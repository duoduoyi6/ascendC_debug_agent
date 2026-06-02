# Precision Tuning Skill — 设计说明

## 概述

修复 AscendC 算子的 **build / import / runtime / timeout / precision** 五类失败。作为 AscendOpGenAgent 中主 agent Phase 8 条件性 spawn 的 debug subagent：主 agent 生成 + 初筛（Phase 0-6）→ Phase 7 `trace-recorder` 判定 `debug_eligible == true` → 进入本 Skill 做深度 debug。

**历史兼容**：原 Skill 只覆盖精度失败，现扩展为五类失败统一入口；精度分支路径（Step 1-P）保持与旧版本兼容。

本 Skill 采用**双 Subagent 架构**，共用同一套基础设施，但提供两种不同的审计策略。

## 文件结构

```
AscendOpGenAgent/
├── agents/
│   ├── ascendc-debug-agent-discovery.md      # 发现式 Subagent（直接从取证 / 日志数据推理）
│   └── ascendc-debug-agent-constructive.md   # 构建式 Subagent（Phase A→B→C 规范化审计）
└── skills/ascendc/ascendc-debug/
    ├── SKILL.md                           # Skill 执行手册（Step 0 ~ Step 4，含 Step 1-P/B/I/R/T + Step 2 Sub-step 2.1~2.6；Step 5/6/7 外置至 exit-protocols.md；编排/循环/退出由 engine/ 引擎掌控）
    ├── README.md                          # 本文件：设计文档
    ├── STRUCTURE.md                       # 目录结构说明
    ├── engine/                            # 事件溯源调试引擎（方案 C：引擎主导主循环）
    │   ├── __main__.py                    # CLI 入口：python -m engine {task_dir} --op-name ...
    │   ├── types.py                       # 闭集类型：FailureType/SessionOutcome/Action/LoopSignal
    │   ├── events.py                      # EventWriter（追加写 events.jsonl）+ read_events
    │   ├── transition.py                  # 写侧：record_* 事件写入函数
    │   ├── state.py                       # 读侧：derive_counters 纯 fold + DebugState
    │   ├── gate_adapter.py                # parse_gate_output + run_gate（subprocess precision_gate.py）
    │   ├── next_action.py                 # debug_next_action：路由+三道闸+loop_signal 派发（纯函数）
    │   ├── runner.py                      # run_debug_session 主循环
    │   ├── exit_artifacts.py              # 从 events 产出 debug_status.json + debug_trace.md
    │   ├── agent_backend.py               # spawn_diagnose_agent（唯一 spawn 点）+ make_agent_callback
    │   └── tests/ut/engine/               # 119 UT（stdlib unittest）
    ├── scripts/                           # 共用脚本
    │   ├── precision_forensics.py         # 数值取证 (L0-L4 + L6 + L8 + available_files)
    │   ├── _forensics_child.py            # 取证子进程 worker（被 precision_forensics.py 以 subprocess 方式调用）
    │   ├── precision_gate.py              # Gate 入口路由器（派发到 gates/ 分支层）
    │   ├── verify_status.py               # verify_status.json loader / validator
    │   ├── precision_knowledge.py         # 知识库管理: load / search / check / dump
    │   ├── anticheat.py                   # 反作弊: wrapper hash + AST + C++ 源码扫描
    │   ├── debug_precision_template.py    # 精度调试分析脚本模板（误差分布 + 固定输入 + shape 二分）
    │   └── gates/                         # 2 层 Gate 包：通用层 + 分支层
    │       ├── __init__.py
    │       ├── common.py                  # 通用层: 反作弊 / AST / baseline / verify_status / 目录完整性
    │       ├── branch_precision.py        # 1-P 分支: F/A/V 精度 Gate (抽自原 precision_gate.py)
    │       ├── branch_build.py            # 1-B 分支: 编译错误 Gate
    │       ├── branch_import.py           # 1-I 分支: import kernel_side 符号 Gate
    │       ├── branch_runtime.py          # 1-R 分支: runtime crash Gate
    │       └── branch_timeout.py          # 1-T 分支: 死锁 / 死循环 Gate
    └── references/                        # 共用参考资料
        ├── precision_knowledge_base.json  # 精度问题知识库（45 条：40 问题模式 + 5 算子 CHECKLIST）
        ├── branch-build.md                # Step 1-B 编译错误分析（build_failed 分支，SKILL.md 外置）
        ├── branch-import.md               # Step 1-I import 错误分析（import_kernel_side 分支，SKILL.md 外置）
        ├── branch-runtime.md              # Step 1-R 运行时错误分析（runtime_error 分支，SKILL.md 外置）
        ├── branch-timeout.md              # Step 1-T 超时分析（timeout 分支，SKILL.md 外置）
        ├── exit-protocols.md              # 归档 / Step 5 成功 / Step 6 失败 / Step 7 退出产物协议（SKILL.md 外置）
        ├── phase-a-checklist.md           # Phase A [REFERENCE_IMPL_SPEC] + Phase C [KERNEL_STEP_TRACE] 格式模板
        ├── run_precision_debug.sh         # 精度调试脚本运行入口（本地 / 远程 Docker）
        ├── bug_examples/                  # 精度缺陷诊断案例库（5 个）
        │   ├── fp16-no-upcast.md          # FP16→FP32 缺失升精度
        │   ├── gm-offset-error.md         # GM 偏移量公式 sizeof(T) 多余
        │   ├── tail-tile-misalign.md      # 尾 tile 未使用 curTileLength
        │   ├── multicore-tiling-overlap.md # 多核 tiling 区间重叠
        │   └── async-sync-missing.md      # 核内队列同步缺失
        └── decomposition_examples/        # 算子计算分解示例（Sub-step 2.2 参考）
            ├── README.md                  # 分解示例索引
            ├── softmax.md                 # Softmax: 单行归约 (5 步)
            ├── layer_norm.md              # LayerNorm: 单行归约 3-pass (7 步)
            ├── reduce_sum.md              # ReduceSum: 单步归约 + 跨步访存 (2 步)
            ├── mse_loss.md                # MSELoss: 跨核两阶段归约 (6 步)
            ├── matmul.md                  # MatMul: 分块累加 (4 步)
            ├── average_pooling2d.md       # AvgPool2d: 滑窗累加 (3 步)
            ├── cumsum.md                  # CumSum: 前缀累加 (2 步)
            ├── rms_norm.md                # RMSNorm: 单行归约 + 逐元素缩放
            └── flash_attention.md         # FlashAttention: 分块 QKV 注意力计算
```

**上下游依赖**：
- 上游：
  - 主 agent 产出 `{task_dir}/kernel/`、`model.py`、`model_new_ascendc.py`、`trace.md`（含 Phase 7 写入的 `final_status` JSON block）
  - `utils/verification_ascendc.py` + `utils/classify_verify_result.py` 产出 `{task_dir}/.verify_status/latest.json`（及 `phase{N}_attempt{M}.json` / `.verify_logs/phase{N}_attempt{M}.log`）
- 下游：
  - 主 agent Phase 8 spawn 后处理读取 `{task_dir}/debug_trace.md` + `{task_dir}/debug_status.json`（本 skill 退出前强制产物）
  - `utils/verification_ascendc.py`（评测）、`utils/run_ascendc_debug_batch_cc.sh`（批量调度，engine 驱动）
  - `python -m engine {task_dir} ...`（直接调用引擎 CLI）

## 2 层 Gate 架构

| 层 | 文件 | 职责 |
|---|---|---|
| 通用层 | `scripts/gates/common.py` | 所有分支共享的不变量：反作弊 hash / AST 退化 / baseline 存在 / 目录完整性 / `verify_status` 产出 / `.json.bak` 未被破坏 |
| 分支层 | `scripts/gates/branch_*.py` | 每类失败的专属 F/A/V 语义；audit section schema 由各分支自定义（通用层不强制统一 schema） |

`precision_gate.py` 作为入口路由器：先跑通用层 → 通过后按 `verify_status.failure_type` 派发对应 `branch_*.py`。

## 双 Subagent 架构

### 发现式 Subagent (`ascendc-debug-agent-discovery`)

**审计策略**: 直接从数值取证数据出发，运用 AscendC 领域知识推理根因。

**特点**:
- 不强制预读参考示例，依赖 Agent 自身的 AscendC 知识储备
- 快速从 diff 模式锁定嫌疑区域
- 适用场景：Agent 对 AscendC API 规范已有充分了解

### 构建式 Subagent (`ascendc-debug-agent-constructive`)

**审计策略**: 严格遵循 Phase A→B→C 的构建式流程。

**特点**:
- Phase A: 先建规范，再看代码（强制读取 `archive_tasks/` 对应案例 kernel + `ascendc-translator` references）
- Phase B: 读取当前 `{task_dir}/kernel/` 实现
- Phase C: 以 `[REFERENCE_IMPL_SPEC]` 为基准进行结构化对照
- 适用场景：需要严格参照规范进行审计

### 共用基础设施

| 组件 | 类型 | 说明 |
|------|------|------|
| `precision_forensics.py` | 脚本 | L0-L8 数值取证 |
| `precision_gate.py` | 脚本 | Gate 入口路由器（派发到 gates/ 分支层）；loop_signal 产出供 engine 消费 |
| `precision_knowledge.py` | 脚本 | 知识库管理 |
| `anticheat.py` | 脚本 | 反作弊: 禁改 wrapper + 扫 C++ 禁调 `at::<op>` |
| `debug_precision_template.py` | 模板 | 精度调试分析脚本模板（误差分布 + 固定输入 + shape 二分） |
| `run_precision_debug.sh` | 脚本 | 调试脚本运行入口（本地 / 远程 Docker） |
| `precision_knowledge_base.json` | 数据 | 精度问题模式库（45 条：40 问题模式 + 5 算子 CHECKLIST） |
| `bug_examples/` | 文档 | 精度缺陷诊断案例库（5 个典型根因 + 实验定位法） |
| `decomposition_examples/` | 文档 | 算子计算分解示例 |

### 策略对比

| 维度 | 发现式 | 构建式 |
|------|--------|--------|
| **分析起点** | 直接从数值取证数据出发 | 先建立规范基准 |
| **Phase A** | 可选查阅 | 强制读取 lowering 示例 |
| **Reference** | 按需查阅 | 必须产出 `[REFERENCE_IMPL_SPEC]` |
| **优势** | 快速、灵活 | 严谨、系统化 |
| **适用** | 经验丰富的 Agent | 规范要求高的场景 |

## 反作弊机制

精度调优的"作弊"指：通过修改 Python wrapper 或在 kernel C++ 里调用 libtorch 计算 API 来掩盖精度失败，而不是真正修复 AscendC kernel。

**三层检测**（由 `anticheat.py` 实现，`utils/run_ascendc_debug_batch_cc.sh` 在引擎驱动的批处理流程中调用）：

1. **wrapper hash 对比**：调优前 snapshot `model_new_ascendc.py` / `model_new_tilelang.py` 的 sha256，调优后对比；变化 = 作弊（wrapper 本应保持不变）
2. **Python AST 退化检测**：通过 `skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py` 检测 wrapper 4 类退化（无扩展导入 / 未调用 kernel / 混用 torch / 标量 for 循环）
3. **C++ 源码扫描**：扫 `kernel/**/*.cpp|h`，捕获 `at::<非白名单 op>` / `torch::<op>` / tensor 计算方法（`x.cumsum()`/`x.histc()` 等）/ `#include <ATen/ops/*.h>` / 缺失 kernel launch

**违规处理**：恢复 wrapper baseline + 标记 🚨 CHEAT（不重跑，进入人工审查队列）。

**agent 文件**（`agents/ascendc-debug-agent{,-discovery}.md`）有"反作弊约束"硬规则章节，明确唯一可改目录为 `kernel/`。

## 信息层级 (L0-L8)

| 层级 | 信息类型 | 状态 | 实现位置 |
|------|---------|------|---------|
| L0 | PASS/FAIL | ✅ 已实现 | precision_forensics.py |
| L1 | 统计值 + 误差分布 | ✅ 已实现 | DiffAnalyzer |
| L2 | 位置特征 (尾块/维度/边界) | ✅ 已实现 | DiffAnalyzer._tail_analysis / _dimension_analysis |
| L3 | 数值特征 (幅值/NaN/符号) | ✅ 已实现 | DiffAnalyzer._check_magnitude_correlation 等 |
| L4 | 张量切片 (per-index) | ✅ 已实现 | DiffAnalyzer._worst_elements |
| L5 | 中间结果探测 | ✅ Agent 手动插桩 (Phase B+) | Sub-step 2.3 Phase B+，产出 [L5_PROBE]（Gate-A 必填） |
| L6 | 内存布局分析 | ✅ 已实现 | MemoryLayoutAnalyzer |
| L7 | 代码位置映射 | ✅ Agent 手动完成 | Sub-step 2.3 L7 手动映射 (静态推算) |
| L8 | 算子语义 | ✅ 部分实现 | OperatorTypeDetector + 知识库 CHECKLIST |

## 分工原则

| 操作 | 执行者 | 原因 |
|------|--------|------|
| L0-L4 数值取证 | Python | 确定性计算 |
| L6 内存布局 | Python | tensor 属性读取 |
| L8 算子类型检测 + 属性提取 | Python | 规则推断，含 dim/reduction_axis |
| 可用文件检测 | Python | 纯文件 IO |
| Pattern hint 分类 + 语义加权 | Python | 规则化，作为建议 |
| Gate 验证 | Python | 结构化检查，loop_signal 由 engine 消费；Agent 不响应 |
| 知识库 IO + RAG 检索 | Python | 结构化评分，纯文件操作 |
| 反作弊 hash / AST / C++ 扫描 | Python | 确定性检测 |
| 算子计算分解 (Sub-step 2.2) | Agent | 需要理解参考实现语义 |
| AscendC 逐步对照 (Sub-step 2.3) | Agent | 需要理解 C++ 代码结构 |
| 根因诊断 (Sub-step 2.4) | Agent | 需要推理 |
| 修复计划 + 代码修复 | Agent | 创造性工作 |

## 链式 Gate 设计

每个 Gate 不仅检查当前步骤产物，还验证前序步骤是否完成：

```
Gate-F (forensics) → 无前置依赖，检查 attempt 号匹配
Gate-A (audit)     → 前置: forensics 存在且 attempt 匹配
                     检查 10 个必填 section: FORENSICS_SUMMARY, COMPUTATION_DECOMPOSITION,
                     REFERENCE_IMPL_SPEC, KERNEL_STEP_TRACE, L5_PROBE, ROOT_CAUSE,
                     CAUSAL_CHAIN_ANALYSIS, FIX_PLAN, TARGET_FILES, EXPERIMENT_RESULTS
                     attempt > 0 时额外检查: DIRECTION_ASSESSMENT（严格二值"是/否"）
Gate-A (fix)       → 前置: audit 存在
Gate-V (validate)  → 前置: 代码文件存在
```

返回码区分：
- 0: 通过
- 1: 产物不完整（可重试）
- 2: 前置依赖缺失（必须回退补完前序步骤）

## 事件溯源引擎（engine/）

调试主循环由 `engine/` 引擎全权掌控，Agent（Subagent）只负责单轮诊断 + 修 kernel，不自驱循环：

| 组件 | 职责 |
|------|------|
| `engine/__main__.py` | CLI 入口，退出码按 SessionOutcome 8 值映射 |
| `engine/runner.py` | 主循环：emit attempt_started → dispatch Action → gate → emit session_done |
| `engine/next_action.py` | 决策纯函数：路由 + 三道闸（全局/分支/wall-clock）+ loop_signal 派发 |
| `engine/agent_backend.py` | 唯一 spawn 点：`claude --bare -p` 单轮诊断+改 kernel，完成即退 |
| `engine/exit_artifacts.py` | 从 events.jsonl 重建 debug_status.json（10 键）+ debug_trace.md（4 节）|

**三道闸预算**：① 全局 MAX_ATTEMPTS=5 ② 分支硬上限 {precision:5, build/import/runtime/timeout:3} ③ wall-clock timeout（主动 emit terminal 事件）

**漂移 = 同 session 续跑**：failure_type 变化时引擎自动重置更靠前分支计数（`_reset_upstream`），同 session 内切换分支，不终止 session。

**Gate-V loop_signal 含义**（由 `precision_gate.py` 产出，engine 消费，Agent 不响应）：

| 信号 | 引擎行为 |
|------|---------|
| PASS | 触发 Done → 产出 debug_status(success) + debug_trace |
| CONTINUE | emit attempt_started(attempt+1)，按当前 failure_type 路由下一轮 |
| STOP | 触发 Done → 产出 debug_status(stopped_by_gate/loop_limit) + debug_trace |

**防 reward-hack**：backend 只判「claude 进程是否正常完成」；「修没修好」由 runner 随后的 Gate-V 客观判定，Agent 无法通过自报告影响 loop_signal。

**运行方式**：
```bash
cd /path/to/AscendOpGenAgent
PYTHONPATH=skills/ascendc/ascendc-debug python3 -m engine <task_dir> \
    --op-name <op> --agent constructive --npu <N> --workdir $(pwd)
```

## 循环控制（引擎分工）

`loop_signal`（PASS / CONTINUE / STOP）由 `precision_gate.py` 产出，由 `engine/runner.py` 消费并驱动下一步 Action，Agent（Subagent）不响应 loop_signal。详见上方"事件溯源引擎"节及 `engine/next_action.py`。

## 知识库

七字段结构（title / feature / patterns / op_types / reason / fix / type），RAG-ready。包含两类条目：

1. **问题模式**（40 条）: 具体精度问题的 feature/reason/fix，带 `patterns` 和 `op_types` 数组；新增 `op_type` 和 `patterns` 字段支持精确匹配检索
2. **算子 CHECKLIST**（5 条）: 按算子类型的精度检查清单（reduction/pooling/loss/matmul/normalization），`patterns=[]`，`op_types` 为算子类别标识

**字段约束**：
- `patterns`：枚举数组，值域 `tail_spike / uniform_offset / scattered / magnitude_correlated / nan_inf_contamination / dimension_concentration / boundary_concentration / all_wrong`；写入时强制校验，非法值 hard error
- `op_types`：自由字符串数组，无枚举限制；CHECKLIST 条目通过此字段路由
- `feature`：纯自然语言，**禁止**内嵌 `pattern=xxx` / `op_type=xxx` 标签

成功修复后自动追加跃迁条目（仅成功时写入，避免污染）。

### RAG 检索

**方案**：结构化关键词筛选 + 评分排序（不使用向量嵌入）

**设计决策**：
1. aarch64 环境下 FAISS / sentence-transformers 依赖重
2. 当前规模（44 条，预期增长到 100-200 条）不需要向量检索
3. 知识库有独立结构化字段（`patterns` 数组 / `op_types` 数组），精确匹配比语义相似度更可靠
4. Fallback 到全量 load 保底，不会漏掉任何条目

**实现**：`precision_knowledge.py search` 命令

```bash
# 第一次检索 (Sub-step 2.1 完成后)
python3 precision_knowledge.py search \
    --kb-path <path> --op-type <type> --pattern <hint> --top-k 3 \
    --log-path <tuning_dir> --attempt N --call-index 0

# 第二次精化检索 (Sub-step 2.4 开始前, 增加位置特征)
python3 precision_knowledge.py search \
    --kb-path <path> --op-type <type> --pattern <hint> \
    --position <tail/boundary/scattered> --top-k 3 \
    --log-path <tuning_dir> --attempt N --call-index 1
```

**评分逻辑**：
- pattern 命中条目 `patterns` 数组 → 权重 3
- op_type 命中条目 `op_types` 数组 → 权重 2
- type 字段与 pattern→type 亲和性映射匹配 → 权重 1
- position 与 pattern 亲和性映射匹配（仅第二次检索）→ 权重 1

**返回结构**：
- `matched_entries`: top-K 普通条目（按 score 降序）
- `checklists`: op_type 匹配的 CHECKLIST 条目（不占 K 配额，始终附带）
- `fallback_to_full_load`: 无任何命中时自动全量返回

## 计算分解示例 (`decomposition_examples/`)

Step 2 的 Sub-step 2.2 要求 Agent 将算子的参考实现分解为逐步计算链。`references/decomposition_examples/` 提供 9 个算子的分解示例，覆盖 5 种计算模式：

| 计算模式 | 示例算子 | 关键审计点 |
|---------|---------|-----------|
| 单行归约 | softmax, layer_norm, rms_norm | padding 值、count 对齐、归约维完整性 |
| 跨核归约 | mse_loss | workspace 同步、Phase 2 正确性、分母计算 |
| 分块累加 | matmul | 累加器初始化、分块边界、精度累积 |
| 滑窗累加 | average_pooling2d | 有效面积计算、边界/padding 处理 |
| 前缀累加 | cumsum | 跨 tile 累加器传递、顺序正确性 |
| 多阶段融合 | flash_attention | Online Softmax 稳定性、KV tile 边界、Q/K/V 分块对齐 |

Agent 在 Sub-step 2.2 中：
1. 先读取 `decomposition_examples/README.md` 了解格式和模式分类
2. 查找与当前算子最匹配的示例文件
3. 按示例的粒度标准完成计算分解
4. 每步标注精度风险点和知识库关联

> 发现式 Subagent 可选查阅分解示例，构建式 Subagent 则强制参考。

## 评测链路一致性

精度调优的 tensor 获取方式与 bench 评测 (`utils/verification_ascendc.py`) 保持**语义一致**，但**代码独立演化**：

- 模型加载：`model.py` 的 `Model` + `model_new_ascendc.py` 的 `ModelNew`（通过 `importlib`）
- 输入：`get_input_groups()` 优先，fallback 到 `get_inputs()`
- 初始化：`ModelNew.get_init_inputs()` 优先（candidate 可覆盖 ref 的 init 参数）
- 容差：dtype-specific 动态阈值（float32: atol=1e-4/rtol=1e-4；float16: atol=1e-2/rtol=1e-3；bfloat16: atol=5e-2/rtol=5e-3）；int8 特判 `atol=1.5, rtol=0.0`；由 `utils/verification_ascendc.py` 的 `_infer_dtype_from_value()` 自动推断
- 设备：`ASCEND_RT_VISIBLE_DEVICES` 环境变量

`precision_forensics.py` **复制**（不 import）了 `utils/verification_ascendc.py` 的 tensor 加载辅助函数到 `OperatorExecutor`，以便独立增强（如直接 dump tensor 做 L1-L4 深度分析）而不破坏 bench。语义升级时需同步两处。

## L5 设计决策：Python 自动化放弃，Agent 手动插桩（Phase B+）替代

**背景**: 早期设计希望通过 Python 脚本自动探测 Kernel 内部每个计算步骤的中间输出，定位误差引入步骤。

**Python 自动化评估（放弃原因）**:

| 维度 | 评估结果 |
|------|---------|
| 实现路径 | 需在 kernel 代码中插入额外 GlobalTensor 输出、修改 host TilingFunc 分配 workspace、重新完整编译 |
| 通用性 | 每个算子的中间步骤不同（ReduceMax/ReduceSum/Exp 各自需要独立探针），无法通用 |
| 侵入风险 | 探针修改可能影响 buffer 对齐，改变精度问题表现（Heisenbug 效应） |
| 工程代价 | 相当于为每个算子维护一个 debug 版本，成本高、不可复用 |

**当前实现（Phase B+）**: Sub-step 2.3 的 Phase B+ 步骤实现了 L5 中间结果探测：
- Agent 在 Compute() 函数的 P1（CopyIn 后）、P2（计算中点）、P3（CopyOut 前）插入 printf 探针
- 遵守 R1-R5 核心规则（GetBlockIdx==0 过滤、DeQue 后读取、half/bf16 转 float、阶段标记、仅 dump 首 16~32 个元素），规避 Heisenbug 效应
- 运行 `debug_{op_name}_precision.py` 提取实测中间值，写入 `[L5_PROBE]` section
- 探针完成后恢复 kernel 原始代码，不影响后续编译

**结论**: L5 Python 自动化不可行，已由 Phase B+ Agent 手动插桩实现。`[L5_PROBE]` 是 Gate-A 的 10 个必填 section 之一。`IntermediateProbe` Python 类保留为存根，不再调用。

## 详细中间文件说明

见 `STRUCTURE.md`。

## TODO 接口清单

| 接口 | 类 | 位置 | 状态 | 说明 |
|------|-----|------|-------|------|
| 中间结果探测 | IntermediateProbe | precision_forensics.py | ❌ Python 存根，不调用 | L5 已由 Phase B+ Agent 手动插桩实现，见 Sub-step 2.3 |
| 代码位置映射 | CodeMapper | precision_forensics.py | ❌ 不实现，由 Agent 手动完成 | Sub-step 2.3 L7 手动映射 |

## Debug 过程完整示例（CumSum 精度失败）

以 `CumSum` 算子、`precision_failed`（mismatch_ratio ≈ 99%）为例，展示各文件和机制在一次完整 debug session 中的调用顺序。

### 0. 触发入口

批处理脚本 `utils/run_ascendc_debug_batch_cc.sh` 的 worker 在容器内执行：

```bash
cd /home/user/AscendOpGenAgent
PYTHONPATH=skills/ascendc/ascendc-debug \
  python3 -m engine tasks/cumsum_task \
    --op-name cumsum \
    --agent constructive \
    --npu 0 \
    --workdir $(pwd)
```

进入 `engine/__main__.py::main()`，构造 `agent_callback` 闭包（`agent_backend.make_agent_callback`），然后把控制权交给 `run_debug_session`。

### 1. runner 主循环启动（engine/runner.py）

`run_debug_session` 进入 while 循环，每拍调用 `debug_next_action(state)`：

```
第 0 拍：state.total_attempts=0, current_failure_type="precision_failed"
→ next_action 返回 Continue(next_attempt=0, next_failure_type="precision_failed")
→ transition.record_attempt_started(attempt=0) 写事件到 .debug_events/events.jsonl
```

### 2. attempt 0 — 轮内三步（forensics → diagnose_and_fix → validate）

#### 步骤 1：forensics（取证）— py_action

`next_action` 发现 `completed_steps={}`，返回：
```python
Action(kind="py_action", step="forensics", ...)
```

runner 执行：
```bash
python3 scripts/precision_gate.py \
    --step forensics --op-name cumsum --task-name cumsum_task --attempt 0
```

`precision_gate.py::_dispatch()` 调用链：

```
precision_gate.py → gates/common.py（反作弊/AST/完整性）
                 → gates/branch_precision.py::run_gate_f()
                      └─ precision_forensics.py::PrecisionForensics.run()
                           ├─ OperatorTypeDetector → L8: op_type="reduction"
                           ├─ OperatorExecutor → _forensics_child.py（子进程隔离）
                           │     ├─ 加载 model.py（PyTorch 参考实现）
                           │     ├─ 加载 model_new_ascendc.py（AscendC wrapper）
                           │     └─ 在 NPU:0 上运行，dump ref/cand tensors
                           ├─ DiffAnalyzer → mismatch_ratio=0.989, pattern_hint="all_wrong"
                           └─ 写 precision_tuning/forensics_report_0.json
```

Gate-F 输出：`{"gate":"GATE-P-F","passed":true,"loop_signal":null}`

runner 记录 `action_completed(step="forensics")` 进 events.jsonl。

#### 步骤 2：diagnose_and_fix — 唯一 spawn 点（engine/agent_backend.py）

`next_action` 返回 `Action(kind="spawn_agent", step="diagnose_and_fix")`。

runner 调 `agent_backend.spawn_diagnose_agent()`：

```python
session_id = str(uuid.uuid4())   # 每轮全新 UUID，天然轮次隔离
result_file = task_dir / "_claude_result_attempt0.json"

cmd = [
    "claude", "--bare", "-p",
    "--agent", "ascendc-debug-agent-constructive",
    "--session-id", session_id,     # 新 UUID，不 resume 历史
    "--add-dir", workdir,
    "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep,Skill",
    "--output-format", "json",
    prompt,   # 含 failure_type/attempt/单轮约束覆盖层
]
subprocess.run(cmd, stdout=result_file, ...)   # 阻塞等待
```

**claude 进程内部（Subagent 视角），读 SKILL.md 走构建式方法论：**

```
Step 2.1: precision_knowledge.py search
          --op-type reduction --pattern all_wrong --top-k 3
          → 命中：「归约轴切分破坏局部归约错误」

Step 2.2: 读 decomposition_examples/cumsum.md（构建式强制）
          → 建立前缀累加规范基准

Step 2.3 Phase B: 读 kernel/cumsum.cpp
          → Compute() 中 scan_axis 假设 axis=1，Python transpose 针对 axis=2

Step 2.3 Phase B+: 插桩 P1/P2/P3 探针
          → P2（ReduceSum 后）实测值全零 → 根因定位

Step 2.4: [ROOT_CAUSE] Host tiling 与 Python transpose 约定不一致
          [FIX_PLAN] 修改 kernel/cumsum_tiling.cpp 的 scan_axis 计算
          写 precision_tuning/precision_audit_0.md（含 10 个必填 section）

Step 3/4: 修改 kernel/cumsum_tiling.cpp → cmake 编译 → Gate-A 校验 audit 格式
```

claude 进程退出，写 `_claude_result_attempt0.json: {"is_error":false,"stop_reason":"end_turn"}`。

`_classify_claude_result()` 读文件 → `success=True, claude_state="ok"`。

runner 记录 `action_completed(step="diagnose_and_fix")`。

#### 步骤 3：validate（Gate-V）— py_action

`next_action` 返回 `Action(kind="py_action", step="validate")`。

runner 执行：
```bash
python3 scripts/precision_gate.py \
    --step validate --op-name cumsum --task-name cumsum_task --attempt 0
```

`branch_precision.py::run_gate_v()` 内部：

```
1. 调 utils/verification_ascendc.py 在 NPU 上运行修复后 kernel
2. classify_verify_result.py::build_status()
   → 产出 .verify_status/latest.json + phase8_attempt0.json
3. 对比 baseline (match_rate=0.011%) vs 当前 (match_rate=52%)
   → 有改善，trend 数据点不足触发 stagnant/harmful_regression
   → loop_signal="CONTINUE"
```

Gate-V 输出：`{"loop_signal":"CONTINUE","stop_reason_code":null,...}`

runner 缓存 GateResult，记录 `action_completed(step="validate", result={loop_signal:"CONTINUE"})`。

### 3. 引擎决策进入下一轮（engine/next_action.py）

`debug_next_action(state, gate_result=GateResult(loop_signal="CONTINUE"))`：

```
gate.loop_signal = "CONTINUE" → 落到预算闸
闸 1: total_attempts(1) < MAX_ATTEMPTS(5) ✅
闸 2: branch_attempt("precision_failed")(1) < 5 ✅
→ Continue(next_attempt=1, next_failure_type="precision_failed")
```

runner 记录 `attempt_started(attempt=1)`，进入第二轮。

### 4. attempt 1 — 精度通过（PASS）

同样三步（forensics → diagnose_and_fix → validate），这次：
- `verify_status` 返回 `match_rate=100.0%`
- Gate-V 输出 `loop_signal="PASS"`

`next_action` 走 **gate 终判优先于预算闸** 的分支：
```python
# gate_result.loop_signal = "PASS" → Done，不经过预算闸
return Done(session_outcome="success", reason="Gate-V loop_signal=PASS")
```

### 5. 终态产物（engine/exit_artifacts.py）

runner 收到 `Done` → 从 events.jsonl 纯 fold 重建：

```
events.jsonl（15 条）:
  session_started
  attempt_started(0) → 3×[started/completed] → attempt_started(1)
  → 3×[started/completed] → session_done(success)
```

产出：

**`debug_status.json`**（10 键）：
```json
{
  "session_outcome": "success",
  "session_branch": "1-P",
  "attempts_used": 2,
  "entry_failure_type": "precision_failed",
  "final_failure_type": "precision_failed"
}
```

**`debug_trace.md`**（4 节：Session Summary / Attempt Log / Final Status / Notes）

进程以退出码 `0` 退出，批处理脚本汇总行记 `success`。

### 6. 完整调用关系

```
批处理脚本
  └─ docker exec → python -m engine（容器内）
       └─ engine/__main__.py
            ├─ agent_backend.make_agent_callback()   # 配置 claude spawn 参数
            └─ runner.run_debug_session()             # 主循环
                 │
                 ├─ [每拍] next_action.debug_next_action()   # 纯决策，无副作用
                 │         ├─ 路由白名单 / failure_type 判断
                 │         ├─ gate 终判优先于预算闸（H1 修复）
                 │         └─ 三道闸（全局/分支/wall-clock）
                 │
                 ├─ [Continue] transition.record_attempt_started() → events.jsonl
                 │
                 ├─ [Action: py_action] subprocess precision_gate.py
                 │     └─ _dispatch() → common.py → branch_precision.py
                 │           ├─ Gate-F: precision_forensics.py
                 │           │     └─ _forensics_child.py（子进程）
                 │           │           └─ model.py vs model_new_ascendc.py on NPU
                 │           │           └─ 产出 forensics_report_{N}.json
                 │           └─ Gate-V: verification_ascendc.py
                 │                 └─ classify_verify_result.py
                 │                       └─ .verify_status/latest.json + loop_signal
                 │
                 ├─ [Action: spawn_agent] agent_backend.spawn_diagnose_agent()
                 │     └─ subprocess: claude --bare -p（每轮全新 session_id）
                 │           └─ Subagent 读 SKILL.md 方法论（Phase A→B→C）
                 │           └─ precision_knowledge.py search（知识库检索）
                 │           └─ 修改 kernel/*.cpp + 编译
                 │           └─ 写 precision_audit_{N}.md
                 │           └─ 进程退出，写 _claude_result_attempt{N}.json
                 │
                 ├─ [Done] transition.record_session_done()
                 │
                 └─ exit_artifacts.build_debug_status()
                       └─ 从 events.jsonl 纯 fold 重建
                       └─ 产出 debug_status.json（10 键）+ debug_trace.md（4 节）
```

**关键分工**：引擎（`engine/`）负责"什么时候做什么"，Gate 脚本（`precision_gate.py` + `branch_*.py`）负责"做对了没有"，Subagent（`claude` 进程）负责"怎么修"。三者通过 `events.jsonl` + `verify_status/latest.json` 解耦，每个 attempt 对应一个全新的 agent session（`uuid.uuid4()` 保证轮次隔离）。
