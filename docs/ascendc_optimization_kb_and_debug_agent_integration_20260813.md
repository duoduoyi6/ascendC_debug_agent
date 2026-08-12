# AscendC 通用性能优化知识库与 Debug-Agent 后置优化阶段设计

日期：2026-08-13

## 1. 决策摘要

本次变更把性能优化视为与精度修复并列的问题类型，但执行顺序不是并行改源码，而是：

1. 先完成精度、完整公开集、anti-cheat 和目标编译门禁，冻结一个正确基线；
2. 再进入独立的性能优化阶段；
3. 每个性能候选先复跑完整正确性门禁，再比较同一 case 集上的重复计时；
4. 只有形成 Pareto 改善才接受，否则自动回滚到正确基线。

新增的 canonical 优化 KB 只记录高度抽象、可跨算子复用的机制，不记录或绑定 910B、910C 等硬件型号。榜单筛选所用硬件、原始 job、submission、源码 SHA 和测量值仅保存在审计证据侧车中，不参与 KB 检索，也不限制知识的适用硬件。

需要保留一个不同概念：性能候选验收时必须冻结测量环境。这里的环境哈希只是保证 baseline 与 candidate 在同一实验条件下可比较，不是 KB 条目的硬件标签。

## 2. 关于 910B 与 910C 的边界修正

根据项目 owner 提供的信息，第一名团队和我方都在 910B 上开发、调试，再提交到 CANNBench 的 910C 环境评测。因此，不能把第一名源码描述成“未经验证的 910C 优化经验”，也没有必要按 910B/910C 拆分通用优化知识。

canonical KB 的抽象方式如下：

- 记录“按 UB 生命周期精确预算”“双缓冲重叠搬运与计算”“分层归约减少全局原子”等机制；
- 记录触发特征、根因、修复方法、适用语义和风险；
- 对指令可用性、UB 容量、队列深度等写成通用前置检查，不写某个产品型号；
- 在任何目标环境采用前都重新编译、跑完整正确性并重复计时。

这与通用精度 KB 的原则一致：知识描述根因和修复模式，具体任务负责验证可用性。

## 3. 数据与筛选口径

输入证据：

- 第一名聚合导出包：`CANNBOT-LINGXI-EVO-official-tasks-1.1.0-910c-sol_f82c1d6f66cc6fe6.zip`；
- 我方 `Debug-Agent Claude Code Kimi K3` 的 2026-08-12 实时榜单快照；
- 我方网站提交归档、candidate inventory、不可变 ZIP 与本地源码 SHA；
- 第一名导出包中的 `report.json`、`manifest.json` 和 83 个提交 ZIP。

第一名文件是由 83 个 submission 聚合出的逐阶段 TOP1，不是一棵由单一作者、单一迭代链产生的源码树。每条优化证据必须绑定实际贡献该 operator/stage 的 submission 和 job。

筛选条件为：

1. 相同算子、相同阶段可配对；
2. 第一名该阶段全部 case 通过；
3. 第一名加速比严格大于 1.0；
4. 第一名加速比严格高于我方；
5. 只有同一不可变 artifact 的隐藏阶段也全部通过，才允许作为 canonical KB 的候选证据。

结果：

| 项目 | 数量 |
|---|---:|
| 配对的 operator/stage 记录 | 106 |
| 满足性能筛选的阶段记录 | 39 |
| 涉及算子 | 26 |
| 同一 artifact 有隐藏全通过证据的阶段记录 | 22 |
| 对应 promotion candidate 算子 | 17 |
| 最终提炼的通用 KB 条目 | 13 |
| promoted / supported | 12 / 1 |

源码关联不能只按算子名匹配，必须按榜单 job 追到 submission，再核验不可变 ZIP 或 candidate inventory 中的文件 SHA。

## 4. 我方源码证据的更正

我方 53 个算子都存在实现。早期“只能对照 18/26”的数字不是“8 个算子没有源码”，而是第一版索引器只扫描 task root 的 `submission_record_initial.json`，没有完整扫描 `website_submissions`、不可变 submission ZIP 和 candidate inventory。

扩展索引后，26 个性能筛选算子的精确版本关联结果为：

| 状态 | 数量 | 说明 |
|---|---:|---|
| 三个或更多提交源码文件完整精确匹配 | 24 | 由不可变 ZIP 或 candidate inventory 文件 SHA 证明 |
| 部分精确匹配 | 1 | `MhcSinkhorn` 的 kernel、launch 可匹配，提交时 plugin 文件未在本地同步为相同 SHA |
| 当前榜单版本映射未闭合 | 1 | `QuantMatmul` 有多份实现和历史 ZIP，但本地没有记录把当前榜单 `job_cf45c1fee2cf` / `job_8a8dd50c921b` 绑定到其中某一份，不能猜测 |

因此准确说法是：25/26 至少有精确源码证据，24/26 完整；唯一未闭合项是 `QuantMatmul` 当前榜单版本的 job-to-artifact 映射。它不是缺少实现。

未进入这 26 个算子的其余 27 个任务也有我方实现，只是没有满足“第一名加速比 > 1 且高于我方”的知识提取筛选条件，无需做本轮源码差分。

## 5. canonical KB 内容

文件：`skills/ascendc/ascendc-debug/references/optimization_knowledge_base.json`

| 类型 | 通用机制 | 状态 |
|---|---|---|
| `OPT_UB_RESIDENT_FAST_PATH` | 工作集可驻留 UB 时使用单遍快路径，大形状保留流式回退 | promoted |
| `OPT_ROW_BATCHING` | 多行或多 group 合批，摊薄标量同步和小 DMA | promoted |
| `OPT_PIPELINE_OVERLAP` | 双缓冲和精确事件重叠 MTE2、VEC、MTE3 | promoted |
| `OPT_SPECIALIZED_PATH_ROUTING` | 按 shape、dtype、layout、属性选择少量专用路径 | promoted |
| `OPT_VECTOR_PASS_ELIMINATION` | 消除无操作 vector pass，融合系数、中间量和写回 | promoted |
| `OPT_ALGEBRAIC_REFORMULATION` | 用等价且代价更低的数学形式替换昂贵复合原语 | promoted |
| `OPT_HIERARCHICAL_REDUCTION` | 先局部聚合，再执行少量全局原子或最终归约 | promoted |
| `OPT_IRREGULAR_MEMORY_ACCESS` | 对不规则访存使用并行 load、Gather、预取和窄类型打包 | supported |
| `OPT_UB_BUFFER_REUSE` | 按生命周期复用 UB 缓冲，精确预算后放大 tile | promoted |
| `OPT_ALIGNED_MAIN_LOOP` | 对齐主循环走大块 DataCopy，尾块单独 DataCopyPad | promoted |
| `OPT_CORE_LOAD_BALANCE` | 按真实工作量均衡分核 | promoted |
| `OPT_DTYPE_AWARE_FAST_MATH` | 按输出 dtype 的误差预算选择 fast math 与精确回退 | promoted |
| `OPT_MEASURED_ROLLBACK_POLICY` | 记录负向实验，按 case 回退而非累积所有优化 | promoted |

每条知识均包含：症状特征、原因、修复模式、适用语义、风险、状态、置信度、验证要求和可追溯 evidence。canonical JSON 中不存在 `hardware_scope` 字段。

## 6. 事实、假设与因果边界

源码结构与榜单高分的共现不能单独证明某一行代码带来全部收益。本次采用三层证据：

- canonical：同一不可变第一名 artifact 隐藏全通过，源码中有明确通用机制，且该机制通常由至少两个独立算子支持；
- noncanonical hypothesis：公开集全通过且性能领先，但同一 artifact 缺少隐藏全通过证据；
- counterexample：同一 artifact 的隐藏正确性不完整，保留用于风险分析，不能晋级。

任何条目应用到新任务时仍是假设，只有在冻结正确基线后，通过一次可归因源码变更、完整正确性和重复性能测量，才算对该任务有效。

## 7. Debug-Agent 应用顺序

推荐顺序是“先精度、后性能”，不是先性能，也不是在同一 attempt 同时改两者。

原因：

1. 错误实现的性能没有稳定语义，可能只是少算、漏算或错误路径更快；
2. 同时修改正确性和性能无法判断收益来自哪一处；
3. 精度阶段先给出可冻结 baseline，性能候选失败时可以无损回滚；
4. 性能优化经常改变求值顺序、tile、cast 和并行归约，必须重新做完整正确性验证。

建议状态机：

```text
精度 Debug-Agent
  -> official full 通过
  -> anti-cheat CLEAN
  -> target compile 通过
  -> 冻结正确源码、case 集和重复性能 baseline
  -> optimization coarse retrieval
  -> profiling / bottleneck 分类
  -> optimization fine retrieval
  -> 一次只做一个可归因优化变化
  -> 先跑完整正确性门禁
  -> 再跑同 case 重复计时
  -> Pareto accept 或 rollback
```

两次优化检索的职责：

- coarse：根据算子语义和静态结构给出可能的优化轴，不直接要求改代码；
- fine：根据真实 bottleneck、逐 case 时间、tile/UB/队列信息返回更窄的候选规则。

## 8. 已实现组件

| 组件 | 状态 | 说明 |
|---|---|---|
| 榜单与源码审计器 | 完成 | `utils/analyze_cannbench_optimization_artifacts.py`，不执行第三方代码 |
| KB 构建器 | 完成 | `utils/build_cannbench_optimization_kb.py` |
| canonical 优化 KB | 完成 | 13 条，硬件无关 |
| noncanonical 侧车 | 完成 | 保存公开假设和隐藏不完整反例，不注入 Agent |
| 只读 coarse/fine 检索器 | 完成 | `scripts/optimization_knowledge.py`，支持 `ABLATE_OPTIMIZATION_KB=1` |
| entry/Pareto 纯函数门禁 | 完成 | `engine/optimization_policy.py` |
| 检索与门禁单元测试 | 完成 | 4 + 4 项；包含 canonical KB 硬件无关约束 |
| 自动接入现有 runner | 未启用 | 当前不会改变精度 Debug-Agent 行为 |

## 9. 上线阶段与门槛

Phase 1 已完成：证据审计、hardware-agnostic KB、只读检索、纯函数验收策略和 shadow 工件。

Phase 2 待完成：为 evaluator 增加统一的逐 case 重复计时 adapter、噪声估计、baseline/candidate manifest 和失败回滚工件。完成标志是相同源码重复运行能稳定给出可审计计时分布。

Phase 3 待完成：在独立 opt-in 配置中接入一个有预算上限的后置性能 attempt。它只能在精度门禁全部通过后运行，候选不达 Pareto 门禁必须恢复 baseline。

Phase 4 待完成：使用冻结任务集做受控消融，对比 precision-only 与 precision-then-optimization 的正确率、性能改善率、回退率、token/时间成本。只有收益稳定且无正确性下降时，才考虑默认开启。

“全量上线”不按日期自动发生，而由以下条件共同触发：

- Phase 2 的计时与回滚工件通过测试；
- Phase 3 至少完成一批隔离 shadow/opt-in 任务且没有 baseline 污染；
- Phase 4 显示性能收益超过噪声和新增成本；
- 默认仍允许用 `ABLATE_OPTIMIZATION_KB=1` 和阶段开关复现实验对照。

## 10. 可复现证据

- 配对明细：`experiments/cannbench_four_slot_k3_20260807/optimization_knowledge_extraction_20260813/paired_operator_stage_metrics.json`
- CSV：`experiments/cannbench_four_slot_k3_20260807/optimization_knowledge_extraction_20260813/paired_operator_stage_metrics.csv`
- KB manifest：`experiments/cannbench_four_slot_k3_20260807/optimization_knowledge_extraction_20260813/optimization_kb_manifest.json`
- 非 canonical 侧车：`experiments/cannbench_four_slot_k3_20260807/optimization_knowledge_extraction_20260813/optimization_hypotheses_and_counterexamples.json`
- canonical KB SHA-256：`a5f2586572170827ccdf2675e89629b02b0e552a508b643d2719d55d618748bf`
- 配对明细 SHA-256：`2a5c923486a3fdeee5478e3e890fe4b4a525e3a785a0555a477017670f8818f4`
