#!/usr/bin/env python3
"""Build a conservative AscendC optimization KB from audited CANNBench evidence.

The input comparison is produced by ``analyze_cannbench_optimization_artifacts.py``.
Only records whose immutable first-place artifact passed the hidden stage are
eligible as canonical evidence.  Public-only results and incomplete hidden
results are exported separately as hypotheses/counterexamples.

This script records source locations and hashes; it does not copy implementation
source from the downloaded leaderboard package into the knowledge base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ENTRY_SPECS = [
    {
        "title": "工作集可驻留 UB 时走单遍快路径，大形状保留流式回退",
        "patterns": ["redundant_gm_pass", "ub_underutilization", "reduction_scalar_stall"],
        "op_types": ["normalization", "reduction", "softmax", "rowwise"],
        "feature": "同一行、group 或 plane 在多遍算法中被重复从 GM 读取，且小中形状的完整工作集实际可以放入 UB。",
        "reason": "统一使用多遍流式实现会在可驻留形状上浪费 GM 带宽，并重复执行 cast、归约准备和同步。",
        "fix": "先按 dtype、工作集和临时缓冲精确计算 UB 占用；满足预算时一次搬入并在 UB 内完成统计、变换和写回，不满足时回退到经过验证的分块/多遍路径。驻留判断必须是显式、可审计的 shape gate。",
        "type": "OPT_UB_RESIDENT_FAST_PATH",
        "applicability": "行归约、归一化、softmax，以及需要两遍读取同一输入的算子。",
        "risks": "不能只按输入张量大小估算；队列深度、cast 缓冲、归约 scratch、padding 和事件对象都计入 UB。错误预算会导致编译失败、UB 越界或隐藏形状崩溃。",
        "evidence_specs": [
            ("GroupNorm", "csrc/ops/group_norm/op_kernel/group_norm_kernel.cpp", "4-15,48-52"),
            ("RmsNorm", "csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp", "9-28,74-129"),
            ("Softmax", "csrc/ops/softmax/op_kernel/softmax_kernel.cpp", "24-33,64-83"),
        ],
    },
    {
        "title": "将多行或多 group 合批，摊薄逐行标量同步和小 DMA 开销",
        "patterns": ["scalar_hot_loop", "tiny_dma", "reduction_scalar_stall"],
        "op_types": ["normalization", "quantization", "rotary_embedding", "rowwise"],
        "feature": "每行计算量较小，但每行都独立发起 DMA、归约读回、除模或同步，scalar pipe 和固定启动成本占主导。",
        "reason": "逐行循环让固定控制开销随行数线性增长，也使 MTE2/V/MTE3 难以形成足够粗粒度的流水。",
        "fix": "在 UB 预算内一次处理 R 行或完整 head group；把行归约、scale 生成、广播和写回改为批量向量操作，并预先展开可复用的 cos/sin、gamma 或 scale 表。R 必须随 dtype、行宽和布局自适应。",
        "type": "OPT_ROW_BATCHING",
        "applicability": "小/中等行宽且行间语义独立的量化、归一化、位置编码和逐行激活。",
        "risks": "合批可能增加 padding、广播表和额外 rotary/gamma 载入；必须保留小 R 或单行回退，并按 case 测量而不是假定越大越快。",
        "evidence_specs": [
            ("DynamicQuant", "csrc/ops/dynamic_quant/op_kernel/dynamic_quant_kernel.cpp", "7-32,133-240"),
            ("RmsNorm", "csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp", "12-20,99-129"),
            ("ApplyRotaryPosEmb", "csrc/ops/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_kernel.cpp", "9-25,93-153"),
        ],
    },
    {
        "title": "用双缓冲和精确事件重叠 MTE2、VEC、MTE3，避免宽泛全流水屏障",
        "patterns": ["pipeline_bubble", "broad_barrier", "tiny_dma"],
        "op_types": ["elementwise", "normalization", "softmax", "optimizer", "layout"],
        "feature": "kernel 按 CopyIn -> Compute -> CopyOut 串行执行，或在每个 tile 使用 PIPE_ALL，使搬运和向量计算互相等待。",
        "reason": "单缓冲及过宽同步会排空流水；即使向量算术很少，端到端时间也被 DMA 和队列等待主导。",
        "fix": "为主输入/输出配置 depth=2 队列或奇偶槽，预取下一 tile，并只在真实 RAW/WAR/WAW 依赖边界使用对应 HardEvent。为不能安全重叠或 UB 不足的形状保留单缓冲回退。",
        "type": "OPT_PIPELINE_OVERLAP",
        "applicability": "有连续 tile、搬运和计算可并行的带宽受限或混合受限 kernel。",
        "risks": "事件 ID 必须正确分配和成对释放；错误复用会造成竞态、挂起或偶发错数。双缓冲还会压缩可用 tile 大小，因此必须联合算 UB。",
        "evidence_specs": [
            ("ApplyAdamW", "csrc/ops/apply_adam_w/op_kernel/apply_adam_w_kernel.cpp", "10-10,83-103,113-130"),
            ("GroupNorm", "csrc/ops/group_norm/op_kernel/group_norm_kernel.cpp", "8-15,23-29"),
            ("Softmax", "csrc/ops/softmax/op_kernel/softmax_kernel.cpp", "36-44,139-180"),
            ("ApplyRotaryPosEmb", "csrc/ops/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_kernel.cpp", "38-40,93-153"),
        ],
    },
    {
        "title": "按 shape、dtype、layout 和属性选择专用快路径，并保留正确回退",
        "patterns": ["layout_dependent_access", "dtype_cast_overhead", "scalar_hot_loop", "random_access_latency"],
        "op_types": ["layout", "normalization", "quantization", "softmax", "gather"],
        "feature": "一个通用循环同时覆盖不同布局、行宽、dtype 和属性组合，导致每个 case 都承担分支、cast、随机访问或不适合的 tile 策略。",
        "reason": "算子的瓶颈会随属性矩阵切换：小行可能受固定开销限制，大行受带宽限制，中间轴可能需要 gather，低精度可能需要或不需要 fp32 桥。单一路径很难同时最优。",
        "fix": "用少量互斥、可证明的运行时/tiling key 路由 resident、batched、streaming、contiguous、interleaved 或 dtype-specific 路径；每条快路径写明适用谓词，并让未覆盖组合落到完整正确回退。",
        "type": "OPT_SPECIALIZED_PATH_ROUTING",
        "applicability": "公开 contract 中存在多 dtype、多布局、动态维度或可选属性的算子。",
        "risks": "路径数量会增加验证面；不得依据隐藏 case ID 或未公开输入硬编码。每条路径都必须跑全公开回归和独立 surrogate 边界测试。",
        "evidence_specs": [
            ("ApplyRotaryPosEmb", "csrc/ops/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_kernel.cpp", "42-50,93-153"),
            ("DynamicQuant", "csrc/ops/dynamic_quant/op_kernel/dynamic_quant_kernel.cpp", "89-129,196-240"),
            ("RmsNorm", "csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp", "74-129"),
            ("Softmax", "csrc/ops/softmax/op_kernel/softmax_kernel.cpp", "6-8,36-44"),
        ],
    },
    {
        "title": "消除无操作向量趟并融合标量系数、中间量和写回",
        "patterns": ["excessive_vector_passes", "dtype_cast_overhead", "ub_underutilization"],
        "op_types": ["elementwise", "activation", "optimizer"],
        "feature": "实现包含 coef=1 的 Muls、coef=0 的 Adds、仅为复制的向量指令、每 tile 重复 Duplicate，或本可复用却另开缓冲的中间量。",
        "reason": "逐元素算子常受向量 pass 数和 UB 带宽限制；数学工作量很小，但每个冗余 pass 都完整读写一次 UB。",
        "fix": "在 host/tiling 端预合并常量并生成 pure/skip 标志；使用 Axpy、原位链和生命周期分析复用已死亡缓冲；把缩放融合到最终写回。任何代数重排都先验证 NaN/Inf、舍入顺序和目标 dtype。",
        "type": "OPT_VECTOR_PASS_ELIMINATION",
        "applicability": "elementwise、激活函数、优化器更新和短算术链。",
        "risks": "浮点运算不满足普遍结合律；对 Inf/NaN、相消和 bf16/fp16 rounding 敏感的重排必须保持 golden 的求值顺序。",
        "evidence_specs": [
            ("Exp", "csrc/ops/exp/op_kernel/exp_kernel.h", "15-39,85-107"),
            ("Sigmoid", "csrc/ops/sigmoid/op_kernel/sigmoid_kernel.h", "15-34,57-67,122-159"),
            ("ApplyAdamW", "csrc/ops/apply_adam_w/op_kernel/apply_adam_w_kernel.cpp", "31-49,174-249"),
        ],
    },
    {
        "title": "用等价且硬件友好的代数形式替换昂贵复合原语",
        "patterns": ["expensive_composite_primitive", "excessive_vector_passes", "ub_underutilization"],
        "op_types": ["activation", "elementwise"],
        "feature": "高阶 API 在目标 SoC 上展开为多条软件模拟指令和额外临时缓冲，例如 Ln/Tanh 或带内部栈缓冲的复合激活。",
        "reason": "API 级单调用不等于硬件单指令；软件展开会增加向量趟数、临时 UB 和同步。",
        "fix": "先展开目标 SoC 上的真实 lowering，再寻找减少高代价原语的数学恒等式或短链；逐 dtype 证明截断范围、特殊值和舍入等价，并保留精确路径。",
        "type": "OPT_ALGEBRAIC_REFORMULATION",
        "applicability": "激活函数和可由短向量链表达的逐元素复合函数。",
        "risks": "近似式不能只看普通有限值；必须覆盖 NaN、正负 Inf、subnormal、饱和区和 dtype 转回误差。",
        "evidence_specs": [
            ("Mish", "csrc/ops/mish/op_kernel/mish_kernel.cpp", "4-13,51-65"),
            ("Sigmoid", "csrc/ops/sigmoid/op_kernel/sigmoid_kernel.h", "15-34,122-159"),
        ],
    },
    {
        "title": "先做核内或分块局部聚合，再执行少量全局原子或最终归约",
        "patterns": ["atomic_contention", "reduction_scalar_stall", "tiny_global_writes"],
        "op_types": ["reduction", "segment", "foreach"],
        "feature": "每个输入元素都直接触发 GM 原子加或标量结果写回，造成原子冲突和大量小事务。",
        "reason": "全局同步/原子的成本远高于 UB 内向量归约；当输出桶较少或分块可控时，重复原子写可以在核内合并。",
        "fix": "按 UB 容量建立局部累加器或 partial workspace，先在 tile/core 内归约，再由每核一次原子写或独立 final kernel 合并；为输出桶过大、dtype 不支持原子或冲突低的情况保留直接路径。",
        "type": "OPT_HIERARCHICAL_REDUCTION",
        "applicability": "segment reduction、foreach reduction、直方图式聚合和多核归约。",
        "risks": "改变归约顺序可能影响浮点误差；workspace 初始化、跨核可见性、原子 dtype 支持和 final kernel 顺序都必须验证。",
        "evidence_specs": [
            ("UnsortedSegmentSum", "csrc/ops/unsorted_segment_sum/op_kernel/unsorted_segment_sum_kernel.cpp", "49-121,153-180"),
            ("ForeachNorm", "csrc/ops/foreach_norm/op_kernel/foreach_norm_kernel.cpp", "11-15,61-113"),
        ],
    },
    {
        "title": "不规则访存用批量独立 load、向量 Gather、预取和窄类型打包写回",
        "patterns": ["random_access_latency", "scalar_hot_loop", "subword_store_overhead"],
        "op_types": ["gather", "indexing", "layout"],
        "feature": "索引和数据依赖形成逐元素随机 GM load；int8/fp16 等窄写回还会产生 partial-line 合并开销。",
        "reason": "串行地址依赖无法隐藏 cache miss；窄标量 store 可能占用多个流水槽并触发读改写。",
        "fix": "把相互独立的索引分组展开形成 MLP；对适合的连续条带先 DMA 到 UB 再用向量 Gather；预取下一组索引覆盖 line miss；满足地址对齐时将多个窄输出打包成 word store。每种 dtype/layout 都要有门控回退。",
        "type": "OPT_IRREGULAR_MEMORY_ACCESS",
        "applicability": "Gather、embedding/indexing 和可转成条带局部性的随机访问。",
        "risks": "寄存器压力、额外 ALU 和错误预取可能反而变慢；word-load+shift 等看似减少事务的方案需独立测量，不能默认启用。",
        "evidence_specs": [
            ("Gather", "csrc/ops/gather/op_kernel/gather_kernel.cpp", "1-35,75-155,228-260"),
        ],
    },
    {
        "title": "按生命周期复用 UB 缓冲并用精确预算换取更大 tile",
        "patterns": ["ub_underutilization", "tiny_dma", "pipeline_bubble"],
        "op_types": ["elementwise", "optimizer", "quantization", "activation"],
        "feature": "多个中间 buffer 的生命周期不重叠却分别分配，导致 tile 被迫缩小、循环次数和队列同步增多。",
        "reason": "UB 是固定容量；冗余 scratch 会同时降低单次搬运粒度和双缓冲可行性。",
        "fix": "画出每个 LocalTensor 的生存区间，只在前值已死亡后复用；按最坏 dtype/path 对每个 queue depth、scratch 和 padding 逐字节核算，留出明确余量后放大 tile。",
        "type": "OPT_UB_BUFFER_REUSE",
        "applicability": "中间算术链较长、cast 缓冲多或多输入/输出的 kernel。",
        "risks": "过早复用会形成隐蔽 WAR/WAW；不同路径的实际缓冲集合不同，预算必须逐路径而不是取平均。",
        "evidence_specs": [
            ("Mish", "csrc/ops/mish/op_kernel/mish_kernel.cpp", "21-34,43-68,92-120"),
            ("ApplyAdamW", "csrc/ops/apply_adam_w/op_kernel/apply_adam_w_kernel.cpp", "44-53,91-103,186-198"),
            ("DynamicQuant", "csrc/ops/dynamic_quant/op_kernel/dynamic_quant_kernel.cpp", "133-195,247-254"),
        ],
    },
    {
        "title": "主循环使用对齐大块 DataCopy，非对齐尾块单独走 DataCopyPad",
        "patterns": ["tail_branch_overhead", "tiny_dma", "alignment_overhead"],
        "op_types": ["elementwise", "reduction", "optimizer"],
        "feature": "每个完整 tile 都执行 count 判断、pad 参数构造或 DataCopyPad，尾块逻辑污染高频主循环。",
        "reason": "大多数 tile 已满足 32B/更高粒度对齐；为罕见尾块支付通用 pad 逻辑会增加 scalar 和 DMA setup 开销。",
        "fix": "将 fullTiles 与 tail 提前拆开；完整 tile 只走固定 count 的 DataCopy，并适度展开主循环；最多一个尾 tile 使用 DataCopyPad。块起点和 tile 长度需按 dtype 对齐。",
        "type": "OPT_ALIGNED_MAIN_LOOP",
        "applicability": "连续 elementwise、归约输入扫描和规则分块的多输入 kernel。",
        "risks": "不能用向上取整的 DataCopy 越过 GM 边界；最后一核、零长度和非整块输入必须走安全尾路径。",
        "evidence_specs": [
            ("ForeachNorm", "csrc/ops/foreach_norm/op_kernel/foreach_norm_kernel.cpp", "85-103,117-163"),
            ("Exp", "csrc/ops/exp/op_kernel/exp_vector_tile.h", "7-24"),
            ("Sigmoid", "csrc/ops/sigmoid/op_kernel/sigmoid_kernel.h", "99-119"),
            ("ApplyAdamW", "csrc/ops/apply_adam_w/op_kernel/apply_adam_w_kernel.cpp", "113-130,141-155,257-266"),
        ],
    },
    {
        "title": "按真实工作量均衡分核，避免余数和三角工作集造成尾核空闲",
        "patterns": ["core_tail_imbalance", "scalar_hot_loop", "pipeline_bubble"],
        "op_types": ["normalization", "reduction", "quantization", "irregular"],
        "feature": "任务数不能整除核数、不同任务成本不等，或连续块划分让部分核提前结束。",
        "reason": "端到端耗时由最慢核决定；平均工作量相同并不代表尾部延迟相同。",
        "fix": "规则等成本任务用 base+remainder 或 stride 分配；不规则任务按可解释的成本函数切分；确保所有已启动核有工作，并记录对齐和局部性权衡。",
        "type": "OPT_CORE_LOAD_BALANCE",
        "applicability": "多行、多 group、多 tile 或任务成本随索引变化的 kernel。",
        "risks": "round-robin 可能破坏连续访问和 cache locality；成本模型必须由 profiling 校准，不能只看任务个数。",
        "evidence_specs": [
            ("GroupNorm", "csrc/ops/group_norm/op_kernel/group_norm_kernel.cpp", "39-51,97-100"),
            ("RmsNorm", "csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp", "55-68"),
            ("DynamicQuant", "csrc/ops/dynamic_quant/op_kernel/dynamic_quant_kernel.cpp", "89-129"),
        ],
    },
    {
        "title": "按输出 dtype 的误差预算选择 fast math，精度不足时保留精确路径",
        "patterns": ["expensive_vector_instruction", "dtype_cast_overhead", "precision_performance_tradeoff"],
        "op_types": ["activation", "elementwise"],
        "feature": "同一近似指令在 fp16/bf16 输出上误差会被最终舍入吸收，但 fp32 输出无法通过精度阈值。",
        "reason": "硬件 Reciprocal、Rsqrt 等通常是估计值；是否可用取决于最终 dtype、误差门限和后续运算，而不是 API 名称。",
        "fix": "逐 dtype 建立误差预算：低精度路径可在全量验证后采用 fast math，fp32 或敏感区间使用 Div/精化迭代；在代码中显式模板分流，不让一种选择覆盖所有 dtype。",
        "type": "OPT_DTYPE_AWARE_FAST_MATH",
        "applicability": "激活、归一化、除法/倒数/平方根密集的逐元素和行归约算子。",
        "risks": "必须覆盖特殊值、接近零分母、subnormal 和边界舍入；不能用平均误差掩盖最坏 case。",
        "evidence_specs": [
            ("Sigmoid", "csrc/ops/sigmoid/op_kernel/sigmoid_kernel.h", "28-34,122-159"),
            ("Mish", "csrc/ops/mish/op_kernel/mish_kernel.cpp", "51-65"),
            ("RmsNorm", "csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp", "12-28"),
        ],
    },
    {
        "title": "把负向性能实验作为一等证据，按 case 回退而不是累积所有优化",
        "patterns": ["performance_regression", "path_overgeneralization", "register_pressure"],
        "op_types": ["all"],
        "feature": "局部上看更少 load、更大对齐或更多预取，但部分 shape/dtype 的端到端性能显著回退。",
        "reason": "优化会改变寄存器压力、tile 尾块、cache locality、DMA 粒度和编译器调度；静态指令数下降不等于真实耗时下降。",
        "fix": "每次只引入一个可归因变化，保存逐 case 基线、候选和噪声区间；仅在正确性不退化且目标 case 集稳定改善时接受。回退结果写入知识条目的 risks/negative_evidence，后续路径路由显式避开。",
        "type": "OPT_MEASURED_ROLLBACK_POLICY",
        "applicability": "所有 performance attempt，尤其是预取、对齐、展开、路径合并和 tile 放大。",
        "risks": "只看聚合均值会掩盖严重 case 回退；同一设备上的一次测量也可能受噪声影响，应使用重复测量和稳健统计。",
        "evidence_specs": [
            ("Gather", "csrc/ops/gather/op_kernel/gather_kernel.cpp", "22-35"),
            ("Mish", "csrc/ops/mish/op_kernel/mish_kernel.cpp", "70-74"),
            ("ApplyRotaryPosEmb", "csrc/ops/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_kernel.cpp", "93-97,123-153"),
        ],
    },
]


FORBIDDEN_CANONICAL_HARDWARE_KEYS = {
    "device_model",
    "hardware",
    "hardware_scope",
    "soc",
    "soc_version",
}


def _assert_hardware_agnostic(value, path: str = "root") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_CANONICAL_HARDWARE_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"hardware-bound canonical KB keys at {path}: {sorted(forbidden)}")
        for key, item in value.items():
            _assert_hardware_agnostic(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_hardware_agnostic(item, f"{path}[{index}]")
    elif isinstance(value, str) and re.search(r"\b910\s*[bc]\b", value, flags=re.IGNORECASE):
        raise ValueError(f"hardware model leaked into canonical KB at {path}")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _knowledge_id(title: str) -> str:
    normalized = " ".join(title.strip().lower().split())
    return "optkb-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _source_file(row: dict, suffix: str) -> dict:
    artifact = row.get("first_place_artifact") or {}
    matches = [item for item in artifact.get("source_files") or [] if item["path"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one source ending {suffix!r} for {row['operator']}, got {len(matches)}"
        )
    return matches[0]


def _evidence(rows: list[dict], operator: str, suffix: str, lines: str) -> dict:
    matches = [
        row for row in rows
        if row["operator"] == operator
        and row["stage"] == "hidden"
        and row.get("knowledge_evidence_tier") == "promotion_candidate"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one hidden promotion candidate for {operator}, got {len(matches)}")
    row = matches[0]
    source = _source_file(row, suffix)
    return {
        "operator": operator,
        "stage": "hidden",
        "first_place_speedup": row["first_place"]["speedup"],
        "ours_speedup": row["ours"]["speedup"],
        "speedup_delta": row["speedup_delta"],
        "first_place_passed_cases": row["first_place"]["passed_cases"],
        "first_place_total_cases": row["first_place"]["total_cases"],
        "first_place_submission_id": row["first_place"]["submission_id"],
        "first_place_job_id": row["first_place"]["job_id"],
        "inner_zip_sha256": row["first_place_artifact"]["inner_zip_sha256"],
        "source_member": f"submissions/{row['first_place']['submission_id']}.zip::{source['path']}",
        "source_sha256": source["sha256"],
        "source_lines": lines,
        "artifact_validation": row["artifact_validation"],
        "local_exact_source_pair_available": bool(row.get("ours_source_candidates")),
    }


def build_kb(comparison: dict) -> list[dict]:
    rows = comparison["rows"]
    entries = []
    for spec in ENTRY_SPECS:
        evidence = [_evidence(rows, *item) for item in spec["evidence_specs"]]
        operators = sorted({item["operator"] for item in evidence})
        status = "promoted" if len(operators) >= 2 else "supported"
        confidence = "high" if status == "promoted" else "medium"
        entry = {key: value for key, value in spec.items() if key != "evidence_specs"}
        entry.update({
            "knowledge_id": _knowledge_id(spec["title"]),
            "status": status,
            "confidence": confidence,
            "validation": {
                "correctness": "every cited immutable artifact passed its 80-case hidden stage",
                "performance": "first-place hidden benchmark speedup is >1.0 and greater than our paired live snapshot",
                "causality": "supported by source structure and leaderboard association, not a controlled one-change ablation",
                "acceptance_requirement": "reproduce on the target task with frozen correctness and repeated same-case timing before adoption",
            },
            "evidence": evidence,
        })
        entries.append(entry)
    _assert_hardware_agnostic(entries)
    return entries


def build_noncanonical_archive(comparison: dict) -> dict:
    rows = []
    for row in comparison["rows"]:
        tier = row.get("knowledge_evidence_tier")
        if tier not in {
            "supported_hypothesis",
            "counterexample_or_rejected",
            "rejected_incomplete_correctness",
        }:
            continue
        rows.append({
            "operator": row["operator"],
            "stage": row["stage"],
            "knowledge_evidence_tier": tier,
            "artifact_validation": row["artifact_validation"],
            "first_place": row["first_place"],
            "ours": row["ours"],
            "speedup_delta": row["speedup_delta"],
            "disposition": (
                "retain for reproduction; do not inject into the general optimization KB"
                if tier == "supported_hypothesis"
                else "retain as a guardrail/counterexample; do not promote"
            ),
        })
    return {
        "schema_version": 1,
        "scope": comparison["comparison_scope"],
        "truth_boundary": (
            "Public-only artifacts and artifacts with incomplete hidden correctness are not "
            "canonical optimization knowledge. Hidden inputs are not inferred."
        ),
        "records": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--kb-output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = _read_json(args.comparison)
    entries = build_kb(comparison)
    _write_json(args.kb_output, entries)
    hypotheses_path = args.evidence_dir / "optimization_hypotheses_and_counterexamples.json"
    _write_json(hypotheses_path, build_noncanonical_archive(comparison))
    manifest = {
        "schema_version": 1,
        "comparison_path": str(args.comparison),
        "comparison_sha256": _sha256_file(args.comparison),
        "optimization_kb_path": str(args.kb_output),
        "optimization_kb_sha256": _sha256_file(args.kb_output),
        "optimization_kb_entries": len(entries),
        "promoted_entries": sum(item["status"] == "promoted" for item in entries),
        "supported_entries": sum(item["status"] == "supported" for item in entries),
        "canonical_scope": (
            "hardware-agnostic mechanisms; snapshot hardware remains only in the audited "
            "comparison sidecar and is not a KB retrieval dimension"
        ),
        "evidence_operators": sorted({ev["operator"] for item in entries for ev in item["evidence"]}),
        "noncanonical_archive_path": str(hypotheses_path),
        "noncanonical_archive_sha256": _sha256_file(hypotheses_path),
        "default_agent_injection": False,
        "reason_default_off": (
            "the current precision engine lacks a same-case repeated-performance baseline and "
            "Pareto acceptance gate; the KB is available through a read-only search CLI"
        ),
    }
    _write_json(args.evidence_dir / "optimization_kb_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
