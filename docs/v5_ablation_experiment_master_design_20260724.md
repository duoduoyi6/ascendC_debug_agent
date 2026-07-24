# V5 消融实验主设计文档

更新时间：2026-07-24
状态：准备阶段，**尚未启动正式实验**

## 1. 研究目标

V5 继承 V4 的可信成功、外部全量验证、反作弊、知识检索、循环控制和可观测性
框架，但使用一批全新的 27 个算子，避免继续围绕 V4 `success16` 调参。实验回答：

1. 完整系统是否提高有完整证据支撑的成功率，而不只是 workflow 局部成功率。
2. KB 只读检索/注入、诊断证据、loop guard、full-eval 和 anti-cheat 各自对应的
   系统级净效应。
3. 关闭能力后，失败是否表现为更多无改善轮次、长上下文、provider 错误、证据缺失
   或不可报告的 wrapper-assisted success。
4. 新增可观测性字段能否完整、语义一致地解释每个终态和成本长尾。

本实验不是组件内部微基准，也不承诺由单次 run 识别所有单组件因果。正式结论必须
由逐算子配对证据、聚合统计和过程日志共同支持。

## 2. 相对 V4 的关键变化

| 项目 | V4 | V5 |
|---|---|---|
| 数据集 | `success16` | 独立收集的 27 个算子 |
| 诊断能力 arm | `no_forensics`、`no_probe` 分开 | 合并为 `no_diagnostic_evidence` |
| LLM | Kimi 系列 | `qwen3.8-max-preview` |
| LLM client context | 随历史模型配置 | 1,000,000 tokens（Claude Code 显式配置，950,000 自动压缩） |
| 执行服务器 | 178 服务器 | `101.245.78.76` |
| 正式 arm 数 | 8 | 7 |

合并诊断 arm 的原因是研究问题关注“整套结构化诊断证据能力”的净作用，而不是分别
估计 probe 与 forensics。该取舍减少一个正式 arm，同时保留系统级对照。

重要边界：

- `no_diagnostic_evidence` 同时设置 `ABLATE_FORENSICS=1` 和
  `ABLATE_PROBE=1`，并设置 `ABLATE_KB=1` 阻断 Agent 手工检索旁路。
- `ABLATE_FORENSICS=1` 会连带移除依赖 forensics 的 `knowledge_search`。
- `diagnose_and_fix` 仍运行，Agent 仍可读取源码、编译日志和运行日志。
- Gate-A、full-eval、loop guard、recovery、anti-cheat 和 turn budget 保持开启。
- 该 arm 只能估计组合净效应，不能声称 probe 或 forensics 的独立因果贡献。

## 3. 数据集与冻结规则

### 3.1 数据源

正式数据集位于：

`/home/wsx/AscendOpGenAgent_assets/cannbot_debug_inputs_n27_20260723/`

启动前必须生成并冻结：

- 恰好 27 个任务的相对路径清单；
- 每个任务输入文件的 SHA256 manifest；
- 算子族、初始 failure type、初始 compact/full-eval 状态；
- 缺失文件、重复任务和不可构建任务审计；
- 只读 source snapshot，所有 arm 从同一 source snapshot 复制任务。
- active 与 backup/full case 数、规范化 case-set SHA256、等价性和 full-eval
  适用性预先登记。

不得因为某个 arm 的运行结果修改源任务。若发现数据本身无效，应在任何正式 arm
启动前修复并重新生成全量 manifest；启动后不得只对个别 arm 换样本。

### 3.2 纳入与排除

纳入条件：

- 具备 `model.py`、待修实现和正式验证所需输入；
- 能在目标容器/NPU 上执行初始验证；
- 初始失败可由统一 failure taxonomy 分类。

启动前必须在隔离副本上对 27/27 算子执行 clean build 和初始验证，并以
round-robin 方式实际覆盖 NPU 3、4、5、6、7。只有 `build_failed`、
`import_failed`、`runtime_error`、
`timeout`、`precision_failed` 五种可调试初始失败可纳入；验证成功、分类器错误和
基础设施不可达均视为 preflight blocker。

若单次 preflight 明确出现 `507015` 或 `NPU_AICORE_EXCEPTION`，最多追加两次
clean-build 复测并完整归档瞬态日志；首个非基础设施结果作为正式初始分类。若三次
均为同类异常，则仍保留 `runtime_error`，不得手工改写为精度失败。

排除条件只允许是数据或基础设施不可运行，不能因为算子困难而排除。所有排除必须在
启动前列入 `dataset_audit.json`，给出原始证据路径。

数据集兼容性补齐记录在 `dataset_amendments.json`。其中缺失的 `model.json`
仅复制同任务当前 `<op>.json`，不改变 case 集或算子实现。

## 4. 正式消融矩阵

| Arm | 关闭内容 | 保持内容 | 核心问题 |
|---|---|---|---|
| `full` | 无 | 全部机制 | 完整系统基线 |
| `no_kb` | KB 检索、注入 | forensics、probe、验证与闸门 | 只读 KB 是否改善成功率、方向选择和成本 |
| `no_diagnostic_evidence` | forensics、依赖它的 KB search、L5 probe | raw-log diagnose、验证与其他闸门 | 整套结构化诊断证据能力是否有净收益 |
| `no_loopguard` | 语义无改善早停和软预算 | 硬预算、验证与其他机制 | loop guard 是否抑制长失败和无效成本 |
| `no_anticheat` | engine 内 anti-cheat | 独立 post-hoc observer | 关闭后暴露多少不可报告或 wrapper-assisted 成功 |
| `no_fulleval` | 运行期 full-eval | compact validation 与其他机制 | workflow 成功缺少多少外部证据 |
| `baseline` | forensics、KB、loopguard、Gate-A、full-eval、probe、recovery | anti-cheat detect-only、硬预算 | evidence-missing/workflow-local 下界 |

`no_probe` 和 `no_forensics` 不再是 V5 正式 profile；wrapper 必须拒绝这两个旧名称，
避免旧 launcher 静默产生语义不同的数据。

`debug_no_audit` 只用于工程诊断，不得进入正式矩阵。

所有启用 KB 的 arm 只允许从同一冻结 snapshot 检索和注入，并强制
`ASCENDC_DEBUG_KB_READ_ONLY=1`；在线写回不属于本次 V5 treatment，也不从
`no_kb` 差异中解释。

## 5. 控制变量

除 arm 指定开关外，以下条件必须固定：

- 同一代码 commit 和代码 SHA256 manifest；
- 同一 27-task source snapshot；
- 同一只读 KB snapshot；
- 同一模型 `qwen3.8-max-preview`；
- 同一 provider endpoint 和 credential 名 `yansong-qwen3-key-1`；
- 同一 Claude Code/Agent 配置、agent spec、allowed tools 和 reasoning effort；
- 同一容器镜像、CANN/PyTorch/编译器版本；
- 同一 NPU 3、4、5、6、7 和五个 worker；
- `max_attempts=5`、`max_turns=240`、`soft_task_turns=480`、
  `max_task_turns=600`、`timeout=43200`，除非 preflight 发现目标环境不兼容；
- 相同任务顺序；若不同 arm 不能并行，则使用预先冻结的 arm 顺序，并记录墙钟窗口。

API key 只保存在 mode `0600` 的服务器 secrets 文件中，不进入 Git、设计文档、
experiment manifest、日志或报告。manifest 仅记录 credential 名和 redacted config。
Qwen endpoint 当前不支持额度查询，因此所有 arm 显式关闭 usage query 和
mixed-provider 调度；单一 `yansong-qwen3-key-1` 固定服务五个并发 worker。运行期
仍记录 400/401/403/429/5xx、refusal 和超时，但不制造或推断额度余额。

## 6. 成功与成本口径

### 6.1 成功分层

每个任务同时报告：

- `Objective Success`：客观验证通过；
- `Workflow Reportable Success`：workflow 判定可报告；
- `Evidence-Backed Success`：objective、full-eval、anti-cheat、AST/降级审计等
  该 arm 应具备的证据均完整且无 wrapper/evidence-missing 风险；
- `Post-hoc Clean Success`：在与 treatment 目录隔离的副本上，由统一 evaluator
  对终态 kernel 执行的外部审查结果。

`completed`、`session_outcome=success` 或 compact pass 均不能单独等价于 clean
success。`no_fulleval`、`no_anticheat` 和 `baseline` 的 workflow success 必须与
evidence-backed success 分开。

### 6.2 主成本与失败 cycle 诊断证据

主成本口径是“每个算子最终有效任务 cycle”的完整消耗：

- 该 cycle 内所有 attempts、same-attempt provider segments、turns、tokens 和成本
  均计入；
- 若 provider/API 故障触发 supervisor 新 cycle 并执行 `reset_target()`，新 cycle
  视为新的算子任务，旧失败 cycle 不并入主成功成本；
- 旧失败 cycle 的 turns/tokens/cost、Claude result、validation、forensics 和
  provider 错误仍完整留档，但只用于 provider stability 与失败根因诊断；
- 不生成“总实验负担”聚合成本，不把失败 cycle 混入 arm 成本均值、成功样本成本、
  paired cost 或任何主结论。

同时报告：

1. Final-valid-cycle turns/input/output/cache tokens/cost；
2. 每个 evidence-backed success 的 tokens/cost；
3. 成功样本交集上的 paired cost，避免不同成功集合造成选择偏差；
4. 失败 cycle 次数和失败类型，仅作为独立诊断表，不汇总为 treatment 成本。

## 7. 指标

### 7.1 聚合指标

| 指标 | 定义 |
|---|---|
| WRSR | `reportable_success / 27` |
| OSR | `objective_success / 27` |
| EBSR | `evidence_backed_success / 27` |
| Post-hoc clean WRSR | 统一外部 evaluator clean pass / 27 |
| RSIR | `(workflow_success - evidence_backed_success) / 27` |
| OSRE | `(objective_success - evidence_backed_success) / 27` |
| CSC | final-valid-cycle total cost / evidence-backed successes |
| Token per EBS | final-valid-cycle total tokens / evidence-backed successes |

成功率比较使用逐算子配对表，并至少给出 paired bootstrap 置信区间或 exact
McNemar 检验；N=27 下不只报告均值和单个 p-value。

### 7.2 机制指标

- Long-failure count：按 turns、tokens、cost 三个分布共同识别；
- No-improvement turns / attempts；
- context-limit、output-limit、max-turn、refusal、400/401/403/429/5xx；
- probe policy compliance、probe metadata completeness；
- direction metadata 和 direction-switch 历史；
- KB retrieved/injected/declared-used ID 完整性；
- forensics build/reuse/cache/degraded evidence；
- checkpoint、current_best、rollback 完整性；
- full-eval 与 anti-cheat evidence completeness；
- SIGTERM/SIGKILL 与 NPU/runtime 基础设施异常。

长失败必须追溯实际 Claude result、events、validation、forensics 和 provider 日志，
不能仅凭最终 `debug_status.json` 推断。

full-eval 覆盖范围按规范化 case 内容而非只按行数判断：

- backup/full case 数大于 active 时必须运行；
- case 数相等但规范化内容或 case-set SHA256 不同时也必须运行；
- 仅当 case 数和规范化 case-set SHA256 都相同才可跳过，并显式记录
  `coverage_equivalent=true`。

## 8. 执行协议

### 8.1 启动前门禁

正式启动必须同时满足：

1. V5 commit、代码 fingerprint 和 remote working tree 一致；启动时重新逐文件
   校验 code/source/KB manifest，拒绝 missing/extra/changed 文件；
2. engine、utils、KB 单元测试全部通过；
3. 27-task dataset audit 通过；
4. 容器、NPU 3-7、CANN、Claude CLI 和 27/27 隔离 evaluator preflight 通过；
5. provider smoke 的实际 `modelUsage` 仅出现 `qwen3.8-max-preview`；
6. provider redacted config、credential file mode 和日志脱敏检查通过；
7. KB snapshot 已冻结为只读并记录 SHA256；
8. 七个 arm 的 launcher/manifest 已生成但没有 worker 进程；
9. `no_diagnostic_evidence` smoke 证明两个开关同时生效，且其他闸门仍开启；
10. 每个 arm 的 post-hoc evaluator 都在隔离深拷贝上运行并重建，不读取 treatment
    的 stale build/runtime 产物，不回写 treatment task；
11. 单 provider、关闭 usage query、五并发和七个 arm 的冻结 manifest 与 launcher
    参数一致。
12. 每个 arm 启动前和 post-hoc 完成后重新校验 code/source/KB fingerprint；发生
    漂移立即停止，不进入下一 arm。
13. provider env 只落在 `.secrets` 下的临时 `0700` 目录，supervisor 退出时删除，
    不写入 arm output。

任一项失败都只记录 blocker，不启动正式实验。

### 8.2 正式顺序

默认按预冻结顺序串行 arm，arm 内多 NPU 并行：

`full -> no_kb -> no_diagnostic_evidence -> no_loopguard -> no_anticheat -> no_fulleval -> baseline`

若 provider 或 NPU 基础设施发生全局中断，保留当前证据并按预定义 retry policy
处理；不得仅重跑结果较差的 arm。provider reset cycle 完整留档但不计入
final-valid-cycle 主成本。

### 8.3 收口条件

每个 arm 必须满足：

- 27/27 具有明确终态；
- supervisor/controller 正常返回；
- 无残留 worker；
- task-level status、validation、provider、turn/token/cost 和 observability 汇总可复算；
- treatment profile compliance 审计通过；
- post-hoc observer 完成且未污染 treatment。

## 9. 分析与因果边界

允许的结论：

- 某 arm 相对 full 的配对净变化；
- 联合诊断证据能力与成功率、长失败或成本之间的稳定关联；
- 某完整证据闸门对报告可信度的必要性。

不允许的结论：

- 从 `no_diagnostic_evidence` 单独归因 probe 或 forensics；
- 从一次 27-task run 推断模型或组件的普遍因果效应；
- 把 provider/NPU 中断算成 agent 能力失败；
- 把缺 full-eval/anti-cheat evidence 的 workflow success 写成 clean success；
- 只在成功任务集合不同的情况下比较平均成本并宣称更省。

若需要拆分 probe 与 forensics 的独立作用，应另建后续 2x2 诊断实验，不回填到
本 V5 主矩阵。

## 10. 预期产物

准备阶段：

- `experiment_control/code_snapshot.sha256.json`
- `experiment_control/provider_config_redacted.json`
- `experiment_control/dataset_manifest.sha256.json`
- `experiment_control/dataset_audit.json`
- `experiment_control/kb_snapshot.sha256.json`
- `experiment_control/environment_snapshot.json`
- `experiment_control/ablation_profile_smoke.json`
- `experiment_control/arm_contract_verification.json`
- `experiment_control/dataset_preflight_results.json`
- `experiment_control/launch_fingerprint_verification.json`
- 每个 arm 前后的 `experiment_control/fingerprints/*.json`
- 七个 arm 的冻结 dry-run manifest

完成阶段：

- 逐任务终态、full/post-hoc evaluator、anti-cheat、provider、observability 数据；
- paired per-task table；
- final-valid-cycle 主成本表和独立的失败-cycle 根因/稳定性表；
- long-failure root-cause table；
- arm compliance/evidence completeness matrix；
- 可复算脚本、原始派生数据和最终 Markdown 报告。
- `utils/analyze_v5_ablation.py` 自动生成的 `v5_analysis.json`、逐任务 CSV、配对
  成功检验、共同成功集合成本、长失败、失败 cycle、arm compliance 和 Markdown
  报告。

## 11. 当前状态

截至 2026-07-24，本阶段只完成代码、数据、provider、环境和 launcher 的准备与
preflight。**未经用户明确指令，不启动任何正式 arm。**
