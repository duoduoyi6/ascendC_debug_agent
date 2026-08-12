# CANNBench A3 榜单逐算子对比

## 快照与范围

- 数据源：[CANNBench 排行榜](https://cannbench.com/leaderboard)
- 抓取时间：2026-08-12 01:01（Asia/Shanghai）
- 筛选条件：`official-tasks`、硬件 `A3`、版本 `1.1.0`、算子与提交标签均为“全部”
- 对比方案：第 1 名 `CANNBOT LINGXI-EVO` 与第 5 名 `Debug-Agent Claude Code Kimi K3`

| 榜单指标 | 第 1 名 | 我方 |
|---|---:|---:|
| 总分 | 7536.41 | 5417.71 |
| 提交算子 | 53/53 | 45/53 |
| 公开测例（榜单头部） | 1049/1060 | 895/900 |
| 隐藏测例（榜单头部） | 4048/4240 | 3235/3360 |
| 整体基准加速比 | 0.972x | 0.342x |
| 聚合明细中的最终生效记录 | 106 条、53 个算子 | 86 条、44 个算子 |

我方榜单头部显示 87 个任务、45 个算子，但展开的“聚合后的算子贡献明细”只有 86 条最终生效记录、44 个算子。差异来自正在评测的 `Scatter`：标准任务 `job_bf98604b20e5` 已 20/20，隐藏任务 `job_b0c9347bf73e` 在抓取时仍处于 performance，尚未形成最终生效明细。因此，下面不把 Scatter 的未终态结果混入比较。

## 比较口径

1. 只比较我方已经有最终生效明细的算子和阶段，共 44 个算子、86 条记录。
2. 其中 42 个算子同时有公开与隐藏结果；`AdaptiveAvgPool3D` 和 `WeightQuantBatchMatmul` 只有公开结果。缺失的隐藏阶段不按 0 分处理。
3. “通过测例数”按同一算子的可比阶段求和，并同时保留公开/隐藏拆分。
4. “加速比”使用明细中的“基准加速比”。公开与隐藏是两个独立值，不做算术平均或其他无榜单依据的合并。
5. `AdaptiveAvgPool3D` 的我方公开任务无 performance 数据，故不参与加速比比较，但其 20/20 仍参与通过测例数比较。

## 我方通过测例更多

共有 **16 个算子**。表中“公开”和“隐藏”均为“我方 / 第 1 名”。

| 算子 | 公开 | 隐藏 | 可比总数（我方 / 第 1 名） | 差值 |
|---|---:|---:|---:|---:|
| AddRmsNormDynamicQuant | 20 / 20 | 80 / 78 | 100 / 98 | +2 |
| Conv2D | 20 / 20 | 80 / 79 | 100 / 99 | +1 |
| Conv3DBackpropFilter | 20 / 20 | 80 / 76 | 100 / 96 | +4 |
| DepthwiseConv2D | 20 / 20 | 80 / 68 | 100 / 88 | +12 |
| DequantSwigluQuant | 20 / 20 | 80 / 78 | 100 / 98 | +2 |
| Gcd | 20 / 20 | 80 / 78 | 100 / 98 | +2 |
| GroupedMatmul | 20 / 14 | 55 / 55 | 75 / 69 | +6 |
| LSTM | 20 / 20 | 61 / 32 | 81 / 52 | +29 |
| Maximum | 20 / 20 | 79 / 78 | 99 / 98 | +1 |
| MhcSinkhorn | 20 / 20 | 80 / 78 | 100 / 98 | +2 |
| NMS | 20 / 20 | 80 / 78 | 100 / 98 | +2 |
| ROIAlign | 20 / 20 | 80 / 79 | 100 / 99 | +1 |
| StridedSlice | 20 / 20 | 80 / 79 | 100 / 99 | +1 |
| TopK | 20 / 20 | 80 / 76 | 100 / 96 | +4 |
| Unique | 20 / 20 | 80 / 78 | 100 / 98 | +2 |
| UnsortedSegmentSum | 20 / 20 | 80 / 77 | 100 / 97 | +3 |

## 我方通过测例更少

共有 **1 个算子**。

| 算子 | 公开（我方 / 第 1 名） | 隐藏（我方 / 第 1 名） | 可比总数（我方 / 第 1 名） | 差值 |
|---|---:|---:|---:|---:|
| EngramGateFusion | 20 / 20 | 9 / 80 | 29 / 100 | -71 |

其余 **27 个算子**通过测例数相同：`AdaptiveAvgPool3D`、`ApplyAdamW`、`ApplyRotaryPosEmb`、`ArgMax`、`CrossEntropyLoss`、`Cummin`、`Dilation2D`、`DynamicQuant`、`Exp`、`ForeachAddcdivScalar`、`ForeachNorm`、`Gather`、`Gelu`、`GridSampler3D`、`GroupNorm`、`MaskedScale`、`Mish`、`MoeFinalizeRouting`、`MoeGatingTopKSoftmax`、`MoeReRouting`、`QuantMatmul`、`RmsNorm`、`Sigmoid`、`Softmax`、`SwiGlu`、`Transpose`、`WeightQuantBatchMatmul`。

在上述 44 个算子的同阶段可比范围内，我方累计通过 **4110** 个测例，第 1 名累计通过 **4107** 个测例，我方净多 **3** 个。该汇总不包含尚未终态的 Scatter。

## 我方基准加速比更高

共有 **5 个算子、6 条阶段记录**高于第 1 名。数值均为榜单显示的基准加速比。

| 算子 | 阶段 | 我方 | 第 1 名 | 差值 |
|---|---|---:|---:|---:|
| EngramGateFusion | 公开 | 0.090x | 0.060x | +0.030x |
| ForeachAddcdivScalar | 公开 | 1.425x | 1.404x | +0.021x |
| ForeachAddcdivScalar | 隐藏 | 4.104x | 4.021x | +0.083x |
| MaskedScale | 隐藏 | 4.295x | 4.233x | +0.062x |
| Maximum | 隐藏 | 3.029x | 2.724x | +0.305x |
| MoeReRouting | 隐藏 | 1.950x | 1.925x | +0.025x |

按算子归类：

- 公开与隐藏均更高：`ForeachAddcdivScalar`
- 仅公开更高：`EngramGateFusion`
- 仅隐藏更高：`MaskedScale`、`Maximum`、`MoeReRouting`

除 `AdaptiveAvgPool3D` 无我方 performance 数据外，共有 85 条阶段记录可比较基准加速比；我方更高 6 条，第 1 名更高 79 条，没有完全相同的记录。因此，我方在通过测例数上已局部超过第 1 名，但性能仍是当前最主要的整体差距。

## 结论

- 正确性方面，我方在 16 个算子上通过测例更多，仅 `EngramGateFusion` 更少；在当前可比范围内总通过数领先 3 个。
- 最大正确性优势来自 `LSTM`（+29）、`DepthwiseConv2D`（+12）和 `GroupedMatmul`（+6）。
- 性能方面，我方仅在 5 个算子的部分阶段领先；只有 `ForeachAddcdivScalar` 在公开、隐藏两个阶段都领先。
- `EngramGateFusion` 是最明确的正确性短板：虽然公开基准加速比略高，但隐藏仅 9/80，不能据此视为整体优于第 1 名。
- 排行榜为动态数据；后续 Scatter 隐藏终态或新的聚合结果生效后，应使用相同筛选和口径重新生成快照。

---

## CANNBench 高加速比算子实现与初始 Seed 来源分析

日期：2026-08-12（Asia/Shanghai）

分析对象：`ForeachAddcdivScalar`、`MaskedScale`、`Maximum`

### 1. 问题与结论

本报告回答四个问题：

1. 三个算子的高加速比来自哪些实现机制？
2. 高加速比是初始 seed 自带、第一次 Debug-Agent attempt 产生，还是后续迭代逐渐累积？
3. campaign 的初始 seed 如何产生，是否随机、是否由某个 Prompt 生成？
4. 当前流程到底是在优化性能，还是只是在修复正确性时偶然得到高性能实现？

#### 1.1 核心结论

1. **高加速比不来自第 2 步放入工程的 seed。** 三个 seed 都是确定性的“输出填零”stub，只保证 schema、输出形状、编译和真实 NPU 启动成立，然后被官方 case 1 判定为 `precision_failed`。它们没有读取输入，也没有实现算子语义，因此不存在可计分的有效性能。
2. **主要性能结构来自 Debug-Agent 生成的第一版真实语义实现。** Agent 虽以正确性修复为目标，但首次实现时就采用了 AscendC vector API、UB tiling、多核切分、队列流水和融合计算，因此产生了较高性能。
3. **三者并不存在统一的“多轮性能累积”模式。**
   - `ForeachAddcdivScalar`：attempt 0 已建立最终计算和 tiling 结构；attempt 1 只修复异步 lambda 生命周期。高性能基本来自第一版真实实现。
   - `MaskedScale`：attempt 0 已建立单 kernel 融合；attempt 1 为通过 fp16 溢出 case，改成更轻的原生 fp16 路径。第二轮既是正确性修复，也可能改善了该 dtype 子集的性能，但没有 attempt 0 的有效官网性能数据，无法量化增量。
   - `Maximum`：第一次可提交候选已取得截图中的隐藏 `3.029x`；后续 int64 正确性修复后的隐藏值为 `3.019x`，反而略低。它不是逐轮累积优化的结果。
4. **当前 campaign 不是性能优化循环。** 配置为 `target_performance=false`、`full_performance=true`：目标 case 调试不跑性能，最终公开全量评测才记录性能；通过条件只要求正确性、完整性和反作弊，不包含最低加速比或“相对上一轮更快”。这些高分是高质量 AscendC 实现的副产品，而非引擎按性能反馈搜索的结果。
5. **截图中的 `Maximum 3.029x` 不是严格终态。** 它对应隐藏 `79/80` 的第一次官网候选。排行榜展示了该候选的性能均值，但该算子当时并未隐藏集严格通过。

#### 1.2 阶段归因总表

| 算子 | 初始 seed | 首次真实语义实现 | 后续关键变化 | 截图值来自哪一阶段 | 判断 |
|---|---|---|---|---|---|
| ForeachAddcdivScalar | 全零输出，不读输入 | attempt 0：实现 `Div -> Muls -> Add`、dtype 路径、双缓冲和 UB-aware tiling | attempt 1：只把异步 lambda 从引用捕获改为值捕获 | attempt 1 打包，但 device 计算结构来自 attempt 0 | 性能主要在第一次真实实现形成 |
| MaskedScale | 全零输出，不读输入 | attempt 0：把 `x * mask * scale` 融合进一个 AscendC kernel | attempt 1：移除 `Abs/Compare/Muls(inf)/Select` 溢出修补，为部分 fp16 组合增加原生路径 | attempt 1 | 主要来自首次融合，第二轮对部分 dtype 有性能影响 |
| Maximum | 全零输出，不读输入 | provider 中断后，recovery attempt 0 先完成普通 kernel，但 host 物化 broadcast 被反作弊拒绝 | recovery attempt 1 改成 device 侧 merged-dim broadcast；隐藏 continuation 再修 int64 | 截图 `3.029x` 来自 recovery attempt 1 的第一个不可变官网候选 | 首次有效候选已经很快；后续正确性修复未提高均值 |

### 2. 证据边界与指标口径

#### 2.1 使用的证据

本报告只使用以下可核查工件：

- 三个 `seeds/<operator>/` 的原始源码；
- task 精确目录中的 `events.jsonl`、`run_summary.json`、`claude_results`、`.cannbench_work/attempt_*`；
- 最终 kernel 源码和 attempt 间源码 diff；
- 官方公开/隐藏 job JSON 中的逐 case `baseline_perf_us`、`elapsed_us`、`speedup`、正确性状态；
- immutable ZIP、source-tree 和 validated-source SHA；
- `prepare_new_public_tasks.py`、`prepare_strict_pass_refill.py`、`agent_backend.py` 与 Debug-Agent 规范。

没有重跑任何已经完成的算子，也没有重新提交官网任务。

#### 2.2 加速比口径

下文使用官网 job JSON 的 `avg_speedup`。在三个严格通过的 job 中，它等于有性能结果 case 的逐 case `speedup` 算术平均；`Maximum` 隐藏失败 case 没有有效 performance，官网均值按其余 79 个有效 case 计算。

公开集和隐藏集即使绑定同一个 submission，也可能得到差异很大的均值，因为 case 的 shape、dtype、规模、属性与 baseline 路径分布不同。**同一 submission 的公开/隐藏差异不能解释成“隐藏阶段又优化了一次”。**

### 3. ForeachAddcdivScalar

#### 3.1 Attempt 与官网结果

| 阶段 | Turns | 结果 | 关键变化 |
|---|---:|---|---|
| seed / attempt -1 | - | CANN9.1 编译通过；官方 case 1 `precision_failed` | 仅对输出做 `Duplicate(0)` |
| attempt 0 | 97 | 单 case 可过，但全量异步运行 `runtime_error` | 从零实现真实算子、dtype 路径、队列和 tiling |
| attempt 1 | 36 | 本地公开 20/20，最终 CLEAN | 仅修复 `RunOpApi(sync=false)` lambda 的 use-after-return |

不可变候选：

- ZIP SHA：`cfa8c4027f4ae31952f5a34dab9c932bdeb9696f0026bd2634e97eb4f97a975a`
- submission：`sub_1b0f5dc0d30a`
- 公开：`job_3dd96e54dfb7`，20/20，`1.4252553296050068x`
- 隐藏：`job_b5ac0ab63a83`，80/80，`4.104287790618998x`

公开和隐藏使用同一个 submission、同一份 ZIP。没有隐藏失败 continuation。

#### 3.2 性能来自哪里

最终 kernel 的核心路径为：

```text
GM x1/x2/x3
  -> DataCopyPad 到 UB
  -> Div(x2, x3)
  -> Muls(..., scalar)
  -> Add(x1, ...)
  -> DataCopyPad 回 GM y
```

主要机制：

1. **复合表达式在一个自定义 kernel 内完成。** 中间 `x2/x3` 和乘 scalar 的结果留在 UB，不写回 GM。
2. **双缓冲流水。** 三个输入队列和一个输出队列深度均为 2，源码见 `foreach_addcdiv_scalar_kernel.cpp:8-9,50-57,142-156`。
3. **按 dtype 专门计算。** fp32 直接运行 `Div/Muls/Add`；fp16/bf16 升到 fp32 后计算，再按对应 RNE 规则降精度，见 `:97-124`。
4. **基于实际 UB 和 AIV 数量生成 tiling。** tile 按队列和 cast 临时区的字节预算计算，限制为 32768 元素并做 64 元素对齐，见 `:224-257`。
5. **TensorList 在同一个异步 OpApi 回调中依次直接 launch。** 最终修复采用值捕获保证所有 tensor 生命周期覆盖 worker 执行，见 `pybind11.cpp:94-136`。

官方公开 case 注释将 baseline 标为 `ForeachAddcdivScalar×1`，因此这里不能简单归因于“把三个 baseline kernel 融成一个”。更准确的解释是：候选直接实现了面向该 contract 的轻量 AscendC 路径，并以定制 tiling、UB 内中间值和较少的通用框架开销取得优势。

逐 case 证据：

- 公开 20 个性能 case 中 18 个 `>=1x`，范围 `0.886x` 到 `2.131x`；
- 隐藏 80 个性能 case全部 `>=1x`，范围 `1.266x` 到 `5.829x`；
- 隐藏 case 38（float32、约 8.39M 输出元素）baseline `524.16us`，候选 `92.36us`，`5.675x`；
- 公开 case 4（float32、约 67.1M 元素）baseline `892.0us`，候选 `957.78us`，仅 `0.931x`。

这说明 `4.104x` 不是一个普适常数，而是同一 kernel 在隐藏 case 分布上取得的平均结果。官网工件没有披露全部隐藏输入规格，因此不能进一步把差异断言为某个确定 shape 或属性。

#### 3.3 是第几轮获得的性能

**主要性能结构在 attempt 0 已经存在。** attempt 1 的 agent 记录明确说明“设备侧 kernel 计算逻辑零改动”，只把 lambda 从 `[&]` 改为按值捕获。最终 ZIP 名为 attempt 1，是因为 attempt 1 才让全量异步执行稳定，并不表示第二轮做了性能优化。

无法给 attempt 0 一个正式官网加速比：它在全量运行中崩溃，不具备有效性能比较资格。但从源码差异和 agent 记录看，最终高性能 device kernel 来自第一次真实实现。

### 4. MaskedScale

#### 4.1 Attempt 与官网结果

| 阶段 | Turns | 结果 | 关键变化 |
|---|---:|---|---|
| seed / attempt -1 | - | CANN9.1 编译通过；官方 case 1 `precision_failed` | 仅对输出做 `Duplicate(0)` |
| attempt 0 | 70 | 19/20；fp16/uint8 大值域溢出 case 失败 | 首次实现单 kernel 融合和 dtype 分发 |
| attempt 1 | 89 | 本地公开 20/20，最终 CLEAN | 为 fp16 + int8/uint8/half mask 增加原生 fp16 路径 |

不可变候选：

- ZIP SHA：`acd58aaa0c4a8a90d54c9c8af814ff05ccef4be594e8d2a8bae2815339f45fd9`
- submission：`sub_592dc5df086f`
- 公开：`job_05daab251897`，20/20，`2.236323413690374x`
- 隐藏：`job_ab248e99e59c`，80/80，`4.294516372760459x`

公开和隐藏仍是同一不可变候选，没有隐藏 continuation。

#### 4.2 性能来自哪里

官方公开 contract 直接标注 baseline 计算通常包含：

- `Cast×1 + Mul×2`；或
- `Cast×1 + Mul×1 + Muls×1`；或
- 同 dtype 时 `Mul×2`。

候选将 `y = x * mask * scale` 放进**单个 AscendC kernel**：每个 tile 只读取一次 x 和 mask，中间结果留在 UB，最后只写一次 y。与多 kernel baseline 相比，它同时减少：

- kernel launch 次数；
- 中间 tensor 分配；
- 中间结果的 GM 写回和再次读取；
- 通用 dtype/cast 调度开销。

其他实现细节：

1. tile 固定为 4096 元素，输入/输出队列深度为 2，见 `masked_scale_kernel.cpp:14-15,45-63`；
2. 编译期分发 15 种 x/mask dtype 组合，避免运行时逐元素分支；
3. int8/uint8 mask 在 dav-c220 上经 half 中转到 fp32，符合硬件 Cast 支持范围；
4. fp16 + int8/uint8/half mask 走原生 fp16 `Mul + Muls`，不分配 fp32 临时区，见 `:17-25,132-170`；
5. 全部尾块由 `DataCopyPad` 处理，保持非对齐 case 的单一路径。

逐 case 证据也符合“融合受益”的解释：

- 公开 20/20 的性能均 `>1x`，范围 `1.044x` 到 `3.207x`；
- 隐藏 80/80 的性能均 `>1x`，范围 `2.498x` 到 `5.883x`；
- 隐藏 case 46（float32、约 6.29M 元素）baseline `262.14us`，候选 `44.56us`，`5.883x`；
- 公开 case 注释明确显示 baseline 为多算子组合，这是三者中证据最直接的高加速来源。

#### 4.3 是第几轮获得的性能

attempt 0 已经完成单 kernel 融合，所以**绝大部分性能优势应在第一次真实实现中形成**。

attempt 1 同时具有性能含义。源码 diff 显示它删除了：

- `Abs`；
- `CompareScalar`；
- 乘 `inf`；
- `Select`；
- 两个额外的 fp32/mask 临时 buffer。

并把相关 fp16 组合改成原生 `Mul + Muls`。因此这轮不仅修复 NaN/inf 位置错误，也缩短了该 dtype 子集的 vector 指令链和 UB 占用。

但不能声称“attempt 1 将加速比从 A 提升到 4.295x”：attempt 0 未通过正确性，没有形成可比较的官网 performance job。现有证据只能证明架构变化方向，不支持定量归因。

### 5. Maximum

#### 5.1 Attempt、官网 cycle 与排行榜值

| 阶段 | Turns | 结果 | 关键变化 |
|---|---:|---|---|
| seed / attempt -1 | - | CANN9.1 编译通过；官方 case 1 `precision_failed` | 仅对输出做 `Duplicate(0)` |
| 初始 attempt 0 | 47 | provider HTTP403，中断 | 只留下半成品 launch 接口，不能计作算子失败或性能结果 |
| recovery attempt 0 | 93 | source anti-cheat 失败 | 完成真实 Maximum，但用 `at::broadcast_tensors(...).contiguous()` 在 host 物化 broadcast |
| recovery attempt 1 | 79 | 本地公开 20/20，最终候选可提交 | 删除 host 物化，改成 device 侧 merged-dim broadcast 和 scalar splat |
| hidden continuation attempt 0 | 48 | 修复 hidden int64 极值 case | int64 从截断到 int32 改为精确 hi/lo 分解 |

第一次不可变官网候选：

- ZIP SHA：`a9d5a1dc4ce9a8c7247f3d093f90172181296235caaafbeba39d25a751fd8806`
- submission：`sub_87e7755d67c8`
- 公开：`job_368c984ed127`，20/20，`0.6740456727451537x`
- 隐藏：`job_cbe5f8e63dc7`，79/80，`3.028728837258591x`

第二次不可变官网候选：

- ZIP SHA：`b542ceddcfda9b505d51b80f914719310c34583ec3dd6bfdf83280e5453be9e1`
- submission：`sub_7add0e9991df`
- 公开：`job_9b5629a508ee`，20/20，`0.6417624118688648x`
- 隐藏：`job_09f3765f0a81`，79/80，`3.019233641186915x`

截图中的 `3.029x` 精确对应第一次候选，而不是后续修复后的候选。第二轮正确性改动后，隐藏均值下降约 `0.0095x`。

#### 5.2 性能来自哪里

Maximum 的关键设计是**不在 host 侧扩展广播 tensor**：

1. host 只对右对齐 shape 做整数运算，将相邻维折叠成最多 8 个 merged dims，生成每个输入的 stride 计划，见 `maximum_kernel.cpp:345-417` 和 `pybind11.cpp:95-103`；
2. inner stride 为 1 时，从 GM 连续搬运；inner stride 为 0 时，只从 GM 读一个 scalar，通过 `GetValue + Duplicate` 在 UB 中展开整 tile，见 `maximum_kernel.cpp:78-123,255-313`；
3. `Max` 在 UB 中直接向量计算，见 `:315-322`；
4. tile 由 UB 预算计算，最大 8192 元素，多核数由输出规模和 AIV 数量决定，见 `:419-470`；
5. fp16/fp32/int32 使用原生 Max，bf16/int8 使用受支持的中间 dtype。

这个设计在广播输入上可避免把一个 scalar 或 size-1 维先扩成完整 GM tensor，理论上能显著降低输入流量和 host 侧通用算子开销。需要注意：官网隐藏工件没有公开每个 case 的完整输入 shape，因此“隐藏高分由广播 case 占比更高导致”只能标为**由实现和计时签名支持的推断**，不能写成已知事实。

逐 case 分布揭示了排行榜均值的局限：

- 公开 20 个 case 只有 1 个 `>=1x`，均值仅 `0.674x`；
- 隐藏 79 个有效性能 case 中 70 个 `>=1x`，官网均值 `3.029x`；
- 隐藏 case 70 只有 1 个输出元素，baseline 有 `10us` floor，候选为 `1.84us`，显示为 `5.435x`；
- 隐藏 case 71 只有 2 个元素，baseline `10us`，候选 `1.96us`，显示为 `5.102x`；
- 隐藏 case 53（float32、约 8.39M 元素）baseline `349.53us`，候选 `64.16us`，显示为 `5.448x`；
- 同时也有很慢的 case，例如隐藏 case 86 为 `0.058x`，说明该实现并非所有广播/规模组合都优。

因此 `3.029x` 既包含真实的大 tensor 路径优势，也受到小 case baseline floor 和隐藏 case 组合的显著影响，不能等价为“Maximum 整体快 3 倍”。

#### 5.3 后续正确性修复为何没有继续涨分

第一次候选把 int64 临时 cast 到 int32 后执行 Max。公开测试的小整数范围能通过，但隐藏 case 67 使用 int64 极值，导致高 32 位丢失。

hidden continuation 为完整 int64 范围增加了 hi/lo 分解、Gather、Compare、Select、多个 mask/offset buffer 和额外事件同步。它修复的是语义正确性，但计算链和 UB 占用都更重。官网隐藏均值从 `3.0287x` 变为 `3.0192x`，与这种代价方向一致。

这条历史直接否定了“每次 Debug-Agent attempt 都会把性能继续累积提高”的假设。

### 6. 第 2 步的初始 Seed 到底如何生成

#### 6.1 通俗版：从官方题目到最终算子源码

可以把一个 CANNBench 算子任务理解为“拿到题目后，从一份故意写错的答案开始改题”：

1. **CANNBench 给题目和考场，不给可提交的正确答案。** 官方提供输入输出契约、公开测试、Python reference/golden、evaluator，以及可以编译扩展的 direct-launch 工程骨架。这些材料说明“算子应该算什么、如何构建和如何评分”，但不是一个可直接提交的正确 AscendC 算子实现。
2. **我先手工写一个可编译但必然算错的 seed。** 本 campaign 的 Codex 控制流程按该算子的输出 shape/dtype 和调用接口编写 `seeds/<operator>/`。它只启动真实 NPU kernel 并把输出全部填成零，不读取输入，也不实现算子公式。它的目的不是提供优化起点，而是稳定制造 `precision_failed`。
3. **准备脚本把三类材料装成一个任务。** `prepare_new_public_tasks.py` 复制官方工程骨架，删除骨架中原有的示例算子目录，放入我们自己的零输出 seed，再附上官方 task contract 和 20 个公开 case 配置。
4. **先证明这个起点确实“能运行但答案错误”。** seed 必须通过 CANN 9.1 编译冒烟，并由官方 evaluator 的真实公开 case 确认 `compile_failed=false`、`precision_failed=true`。如果只是编译失败、没有启动 kernel，就不会交给 Debug-Agent。
5. **Debug-Agent 才开始生成真正的算子实现。** attempt 0 读取官方契约、reference、失败证据、当前 seed 和 AscendC 规范，通常保留可用的工程/launch 骨架，但重写或扩展 kernel、tiling 和绑定代码，实现真正的算子语义。它不是从 CANNBench 官方正确源码复制，也不是随机生成。
6. **后续 attempt 继续修边界问题。** 若公开或隐藏 case 暴露 dtype、shape、广播、对齐、数值极值或生命周期问题，continuation 从上一份不可变候选继续修改；全量正确性、反作弊和目标编译全部通过后，才形成最终提交源码。

各部分的来源和作用如下：

| 部分 | 来源 | 作用 | 是否包含最终 AscendC 算法 |
|---|---|---|---|
| direct-launch 工程模板 | CANNBench 官方 | 提供构建、打包和调用骨架 | 否 |
| task contract、公开 case、reference/golden、evaluator | CANNBench 官方 | 定义语义并验证结果 | reference 描述正确语义，但不是提交用 NPU 源码 |
| `seeds/<operator>/` 零输出 seed | 本 campaign 中由 Codex 手工编写 | 提供可编译、可启动、确定错误的受控起点 | 否 |
| attempt 0 的 first semantic candidate | Debug-Agent | 第一次实现真实 AscendC 算法 | 是 |
| 后续 attempt/hidden continuation | Debug-Agent | 修复公开或隐藏边界问题 | 是，在上一候选上继续演化 |

因此，本报告中的“初始 seed”“第一次真实实现”和“最终提交源码”是三个不同对象。最终算子源码主要由 Debug-Agent 在 attempt 0 及后续 continuation 中生成；CANNBench 官方提供的是题目、reference、测试与工程模板，不提供我们提交的正确 AscendC 实现。

#### 6.2 不是随机生成，也没有 seed-generation Prompt

可执行流程中没有在任务启动时调用模型生成 C++，也没有使用随机数构造源码。`prepare_new_public_tasks.py` 的行为是确定性的：

1. 从官方 `examples/direct_launch_example` 复制工程模板，见 `:115-126`；
2. 从本 campaign 仓库中复制预先由 Codex 手工编写的 `seeds/<operator>/` 零输出失败源码；它不是 CANNBench 官方算子实现，见 `:119,156-167`；
3. 复制官方 task contract，见 `:170-171`；
4. 写入 provenance，明确标记 `candidate_kind=intentionally_incomplete_direct_launch_seed`，见 `:172-185`；
5. 配置官方 20 case、目标 case 1 和后置 target compile，见 `:191-225`。

`prepare_strict_pass_refill.py` 还会在持有 campaign 顺序约束时检查：

- 不覆盖已有 run root，见 `:995-996`；
- frozen inventory 和首个未启动任务未漂移，见 `:1000-1028`；
- 上一槽位确实严格通过并有 immutable SHA，见 `:1030-1059`；
- 然后才调用 `_prepare_task`，见 `:1061-1066`。

三项任务的现场证据一致：seed compile smoke `rc=0`，随后官方 case 1 的 `verification_exit_code=1`、`failure_type=precision_failed`、`compile_failed=false`。

#### 6.3 Seed 的“技巧”是什么

seed 不是高质量实现，而是高稳定性的**失败夹具**：

1. host meta 按官方 schema 分配正确 shape/dtype 的输出；
2. device kernel 将输出按 `uint16_t` halfword 视图处理；
3. 使用 `Duplicate(0)` 在 UB 生成零，再用 `DataCopyPad` 写回；
4. 使用实际 AIV 数量和 UB 大小做一个简单切分，避免因为明显的 launch/tiling 错误先失败；
5. 完全不读取输入、不实现算子语义，确保官方 evaluator 得到真实 `precision_failed`。

三份 seed 的核心代码几乎同构：

- `seeds/foreach_addcdiv_scalar/op_kernel/foreach_addcdiv_scalar_kernel.cpp:9-48`
- `seeds/masked_scale/op_kernel/masked_scale_kernel.cpp:9-47`
- `seeds/maximum/op_kernel/maximum_kernel.cpp:9-46`

这是一种实验控制设计：让不同算子都从“可编译、可启动、确定错误”的状态进入 Debug-Agent，而不是给 Agent 一个来自其他方法的接近成功实现。

配置中的 `target_eval_seeds=[0,1,2]` 是 evaluator 的重复输入随机种子，用于验证修改后的实现，不参与 C++ seed 源码生成。

#### 6.4 真正产生“初始解”的是 Debug-Agent attempt 0

需要区分两个概念：

- **campaign seed**：零填充失败夹具；
- **first semantic candidate**：Debug-Agent attempt 0 第一次实现真实算子语义后的源码。

Debug-Agent 的 prompt 也不是“请随机写一个实现”。`agent_backend.py:_build_prompt` 会确定性注入：

- task 目录、算子名、failure type、attempt 编号；
- 当前反作弊历史、rollback/direction history；
- 通用精度 KB 检索结果；
- CANNBench 公开失败证据与只允许修改 `kernel/` 的边界；
- 单轮修复和 attempt metadata 约束。

系统 Agent 规范要求 precision 分支先做 Phase A：读参考案例和 AscendC API 资料，建立 TQue/TBuf、关键 API、对齐与禁用模式规范；再读当前 kernel，做 Phase B/C 对照后修复。对应代码见：

- `skills/ascendc/ascendc-debug/engine/agent_backend.py:639-674`
- `agents/ascendc-debug-agent-constructive.md:35-90,135-164`

因此第一版真实实现的来源是“官方 contract + 当前失败源码 + 结构化取证 + AscendC 参考规范 + Agent 推理”，不是随机采样，也不是从其他 Agent 的成功实现复制。

#### 6.5 一个值得修正文案的细节

`_benchmark_backend_constraint` 的通用提示写着“当前候选是此前提交到 CANNBench 的精确源码快照”（`agent_backend.py:192`）。这对 hidden continuation 正确，但对全新 seed 任务不严格准确。任务内的 `PROVENANCE.json` 已正确声明它是失败 seed，Agent 也实际识别出了 zero-fill stub；不过后续可把 prompt 按 `candidate_kind` 分成：

- `initial_failure_seed`；
- `immutable_submission_continuation`。

这会减少概念混淆，但不改变本次三个候选的源码或既有结果。

### 7. 对“初始解质量决定加速比”的判断

这个判断需要拆成两部分：

1. **如果“初始解”指第 2 步的 zero-fill seed：不成立。** 它没有算子算法，不能贡献最终性能。它的质量只影响实验是否能稳定进入 precision 分支，例如 schema、输出 shape、编译和 launch 是否正确。
2. **如果“初始解”指 Debug-Agent attempt 0 产生的 first semantic candidate：基本成立。** 三个案例的主要架构选择都在第一次真实实现或第一次有效 recovery 中形成，后续轮次大多是在修正确性、生命周期或反作弊问题。

但当前流程没有保存“每个尚未正确的 attempt 的公平性能分数”，所以不能绘制可靠的 attempt-by-attempt 加速曲线。对错误候选强行计时也可能把崩溃、未执行 kernel、错误输出或反作弊实现当成性能优化，方法上不成立。

### 8. 对后续流程的建议

在不改变当前正确性优先 campaign 的前提下，建议把性能观察做成独立 shadow 阶段：

1. 继续保留 zero-fill seed，不把它包装成“初始优化解”；
2. 当某个 attempt 首次达到公开 20/20、anti-cheat CLEAN 后，保存该 candidate 的公开逐 case performance snapshot；
3. 后续正确性 continuation 若再次达到同样门禁，自动生成与上一有效 candidate 的 paired case delta；
4. 不允许错误候选、反作弊失败候选或 provider 中断候选进入性能趋势；
5. 只有在正确性已经锁定后，才新增一个显式 performance objective；不能让性能目标改变当前三次官网失败、SHA 去重和隐藏提交规则；
6. 对小 case 单独报告绝对延迟和 baseline floor，避免把 `10us -> 2us` 的 5x 与大 tensor 的带宽收益混成同一种结论；
7. 对公开/隐藏差异只记录观察和可证伪假设，不反推隐藏 case 的确切 shape/dtype。

### 9. 最终回答

- `ForeachAddcdivScalar`：高性能结构由第一次真实 Debug-Agent 实现产生；第二轮只修 runtime 生命周期。
- `MaskedScale`：高性能主要来自第一次单 kernel 融合；第二轮的 fp16 原生路径兼具正确性修复和潜在性能收益，但没有可量化的 attempt 0 官网对照。
- `Maximum`：截图高分来自第一次有效不可变候选；后续正确性修复没有提高性能均值。其隐藏高分受 case 分布、广播路径和小 case baseline floor 共同影响，且该候选仅 79/80。
- 第 2 步 seed 是确定性的 zero-fill failure fixture，不是随机实现、不是 Prompt 生成，也不是高加速比来源。
- 当前 campaign 测量性能但不以性能驱动迭代；这些高加速比是 Debug-Agent 首次构造的高质量 AscendC 数据流与算子融合带来的涌现结果。

### 10. 主要证据索引

#### 流程与 Prompt

- `prepare_new_public_tasks.py:103-230`
- `prepare_strict_pass_refill.py:985-1085`
- `skills/ascendc/ascendc-debug/engine/agent_backend.py:174-202,639-674`
- `agents/ascendc-debug-agent-constructive.md:35-90,135-175`
- `utils/verification_cannbench.py:472-620`

#### ForeachAddcdivScalar

- `runtime_evidence/foreach_addcdiv_scalar_refill_from_moe_re_strict_pass_20260810_221330/tasks/023_ForeachAddcdivScalar/`
- `website_submissions/foreach_addcdiv_scalar_attempt1_20260810_234600/`

#### MaskedScale

- `runtime_evidence/masked_scale_refill_from_foreach_norm_strict_pass_20260811_033700/tasks/026_MaskedScale/`
- `runtime_evidence/masked_scale_refill_from_foreach_norm_strict_pass_20260811_033700/experiment_control/submissions/masked_scale_attempt1_20260811_095337/`

#### Maximum

- `runtime_evidence/maximum_refill_from_cummin_strict_pass_20260811_202131/`
- `runtime_evidence/maximum_provider_recovery_20260811_205737/tasks/040_Maximum/`
- `runtime_evidence/maximum_hidden_failure_continuation_20260811_230710/tasks/040_Maximum/`
- `website_submissions/maximum_attempt1_20260811_224314/`
- `website_submissions/maximum_attempt0_20260812_000839/`

---

## 已提交算子的初始精度、榜单精度与累计投入

本节继续使用本文开头冻结的同一榜单快照：2026-08-12 01:01（Asia/Shanghai），筛选条件为 `official-tasks`、`A3`、`1.1.0`。范围是榜单头部计入我方“已提交算子”的 45 个算子，包括当时隐藏评测尚未结束的 `Scatter`。

### 11. 统计口径

1. **初始精度**取正式 campaign 中该算子最早的、属于当前 task 本身的 `baseline_checkpoint`。绝大多数新任务从确定性的 zero-fill seed 开始，因此这里只执行官方 case 1，结果为 `0/1`；这不是完整公开集正确率。
2. **榜单快照精度**按“公开通过数/20 + 隐藏通过数/80”展示。初始 case 1 与最终公开/隐藏不是相同测试集合，因此本节不计算误导性的百分点增幅。
3. **Debug-Agent attempt**按唯一模型 `session_id` 计一次实际 Agent 调用。不同 continuation 会重新从 `attempt0` 编号，个别同一 engine attempt 也可能重新派发 Agent；直接相加目录中的 attempt 编号会漏计或重计，所以使用 session ID 去重。
4. **中间精度轨迹**中的 `A1...An` 是按时间排序后的实际 Agent 调用序号，不是各 continuation 内会重复出现的原始 `attempt0/1`。每个结果只读取同一 task 事件流中紧随该 `debug_worker` 的 `precision_gate.result.objective_validation`，优先使用 `full_eval.passed_cases/total_cases`。
5. 轨迹保留当时 evaluator 的真实分母，不强行归一到 20。例如 QuantMatmul 早期正式快照有 `26/26`。`provider_error`、`未形成候选`、构建失败和 source/CPP/anti-cheat 门禁中止均明确标记“未验证”或“未执行精度”，不伪造成 `0/20`；运行了验证但没有归档 full 逐 case 数时写“无 full 计数”。
6. **Agent token**来自每个 session 的 `modelUsage`，列顺序为“输入 / 输出 / cache-read / 处理总量”。处理总量还包含 cache-creation；本批记录的 cache-creation 均为 0。它表示模型处理的上下文规模，不等同于未缓存 token，也不能直接当作账单成本。
7. attempt 后括号内是报告了非零 token usage 的 session 数。共有 4 个一轮即失败的 provider session 报告 0 token，仍计入实际 Agent 调用次数。
8. **CANNBench 测试次数**按唯一 `job_id` 计数，并拆分标准/隐藏。只有服务端实际生成 job 的请求才计入；本地 evaluator、没有生成 job 的网络/provider/配额失败和同一 job 的重复证据副本不计。
9. 对 36 个严格通过算子，累计截止其严格通过版本；对待定、未严格通过、平台阻塞和隐藏在途的 9 个算子，累计截止本榜单快照，并不虚构“最终通过版本”。
10. campaign 范围只包含 2026-08-06 的 public20 正式批次、2026-08-07 起的主 campaign，以及 LSTM 退役同事 61-case 后的官方 public/hidden 流程。7 月消融实验和 LSTM 旧 61-case 均不计入。

`GroupedMatmul` 是特殊情况：当前可审计 campaign 的最早 checkpoint 已经是公开 `20/20`，说明本次归档从一个公开集已成功候选开始；没有证据支持把它改写成 zero-fill `0/1`。LSTM 的官方流程则从真实 `build_failed` 开始，未执行精度，因此不写成 `0/1`。

### 12. 逐算子明细

| 算子 | 初始可审计验证 | 每次 Debug-Agent attempt 后的本地官方验证 | 榜单快照（公开 + 隐藏） | 快照状态 | Debug-Agent attempts（其中有非零 token usage） | Agent tokens：输入 / 输出 / cache-read / 处理总量 | CANNBench jobs：标准 + 隐藏 = 总计 |
|---|---:|---|---:|---|---:|---:|---:|
| AdaptiveAvgPool3D | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 未运行 | 平台阻塞 | 1（1） | 131,788 / 64,171 / 5,420,544 / **5,616,503** | 1 + 0 = **1** |
| AddRmsNormDynamicQuant | 0/1 case 1（precision_failed） | A1: cpp_regression_gate（未执行精度）<br>A2: provider_error（未验证）<br>A3: provider_error（未验证）<br>A4: 19/20（precision_failed）<br>A5: provider_error（未验证）<br>A6: 20/20 | 20/20 + 80/80 | 严格通过 | 6（5） | 364,178 / 225,469 / 9,012,736 / **9,602,383** | 1 + 1 = **2** |
| ApplyAdamW | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 97,205 / 49,161 / 2,182,144 / **2,328,510** | 1 + 1 = **2** |
| ApplyRotaryPosEmb | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 151,845 / 88,516 / 9,283,328 / **9,523,689** | 1 + 1 = **2** |
| ArgMax | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: provider_error（未验证）<br>A3: provider_error（未验证）<br>A4: 20/20 | 20/20 + 80/80 | 严格通过 | 4（4） | 726,842 / 467,201 / 49,361,920 / **50,555,963** | 1 + 1 = **2** |
| Conv2D | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: 2/20（precision_failed）<br>A3: provider_error（未验证）<br>A4: 20/20 | 20/20 + 80/80 | 严格通过 | 4（4） | 461,971 / 237,051 / 21,099,520 / **21,798,542** | 1 + 1 = **2** |
| Conv3DBackpropFilter | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: provider_error（未验证）<br>A3: 19/20（precision_failed）<br>A4: provider_error（未验证）<br>A5: 20/20<br>A6: 20/20 | 20/20 + 80/80 | 严格通过 | 6（5） | 589,838 / 375,533 / 26,035,200 / **27,000,571** | 2 + 2 = **4** |
| CrossEntropyLoss | 0/1 case 1（precision_failed） | A1: build_failed（未执行精度）<br>A2: 18/20（precision_failed）<br>A3: provider_error（未验证）<br>A4: provider_error（未验证）<br>A5: 20/20<br>A6: 20/20 | 20/20 + 77/80 | 未严格通过 | 6（6） | 524,969 / 245,141 / 11,651,328 / **12,421,438** | 2 + 2 = **4** |
| Cummin | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 258,454 / 132,229 / 8,924,160 / **9,314,843** | 1 + 1 = **2** |
| DepthwiseConv2D | 0/1 case 1（precision_failed） | A1: 18/20（precision_failed）<br>A2: provider_error（未验证）<br>A3: provider_error（未验证）<br>A4: 18/20（precision_failed）<br>A5: 18/20（precision_failed）<br>A6: provider_error（未验证）<br>A7: 18/20（precision_failed）<br>A8: 20/20<br>A9: 20/20 | 20/20 + 80/80 | 严格通过 | 9（8） | 890,187 / 438,583 / 45,094,656 / **46,423,426** | 2 + 2 = **4** |
| DequantSwigluQuant | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: cpp_regression_gate（未执行精度）<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 374,037 / 205,520 / 14,340,096 / **14,919,653** | 1 + 1 = **2** |
| Dilation2D | 0/1 case 1（precision_failed） | A1: 未形成候选（未验证）<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 301,870 / 162,988 / 18,320,201 / **18,785,059** | 1 + 1 = **2** |
| DynamicQuant | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 128,476 / 70,742 / 3,524,864 / **3,724,082** | 1 + 1 = **2** |
| EngramGateFusion | 0/1 case 1（precision_failed） | A1: 20/20<br>A2: 20/20<br>A3: 20/20 | 20/20 + 9/80 | 待定 | 3（3） | 507,381 / 257,376 / 21,678,848 / **22,443,605** | 3 + 3 = **6** |
| Exp | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 112,716 / 51,996 / 4,310,016 / **4,474,728** | 1 + 1 = **2** |
| ForeachAddcdivScalar | 0/1 case 1（precision_failed） | A1: 0/20（runtime_error）<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 192,944 / 84,823 / 7,917,568 / **8,195,335** | 1 + 1 = **2** |
| ForeachNorm | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 287,867 / 153,576 / 33,099,008 / **33,540,451** | 1 + 1 = **2** |
| Gather | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: 20/20<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 241,670 / 139,275 / 6,402,816 / **6,783,761** | 2 + 2 = **4** |
| Gcd | 0/1 case 1（precision_failed） | A1: 20/20<br>A2: 19/20（precision_failed）<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 251,132 / 128,935 / 6,145,536 / **6,525,603** | 2 + 2 = **4** |
| Gelu | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 286,624 / 177,938 / 17,871,616 / **18,336,178** | 1 + 1 = **2** |
| GridSampler3D | 0/1 case 1（precision_failed） | A1: 20/20<br>A2: provider_error（未验证）<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 298,283 / 152,466 / 12,268,800 / **12,719,549** | 2 + 2 = **4** |
| GroupNorm | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: cpp_regression_gate（未执行精度）<br>A3: 20/20<br>A4: 20/20 | 20/20 + 80/80 | 严格通过 | 4（4） | 535,071 / 292,707 / 30,546,688 / **31,374,466** | 1 + 1 = **2** |
| GroupedMatmul | 20/20 public（success） | A1: 20/20<br>A2: 20/20<br>A3: 20/20<br>A4: provider_error（未验证）<br>A5: 20/20<br>A6: 20/20<br>A7: runtime_error（无 full 计数）<br>A8: 20/20<br>A9: provider_error（未验证）<br>A10: 20/20<br>A11: 20/20<br>A12: provider_error（未验证）<br>A13: 20/20<br>A14: 20/20<br>A15: 20/20<br>A16: 20/20<br>A17: 20/20 | 20/20 + 55/80 | 待定 | 17（17） | 3,687,008 / 2,031,847 / 181,388,765 / **187,107,620** | 13 + 1 = **14** |
| LSTM | build_failed（未执行精度） | A1: 18/20（precision_failed）<br>A2: 20/20<br>A3: 20/20<br>A4: 20/20<br>A5: provider_error（未验证）<br>A6: 20/20 | 20/20 + 61/80 | 待定 | 6（6） | 1,302,401 / 674,190 / 118,435,072 / **120,411,663** | 3 + 3 = **6** |
| MaskedScale | 0/1 case 1（precision_failed） | A1: 19/20（precision_failed）<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 249,687 / 131,292 / 9,612,544 / **9,993,523** | 1 + 1 = **2** |
| Maximum | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: cpp_regression_gate（未执行精度）<br>A3: 20/20<br>A4: 20/20 | 20/20 + 79/80 | 未严格通过 | 4（4） | 445,431 / 237,730 / 14,861,824 / **15,544,985** | 2 + 2 = **4** |
| MhcSinkhorn | 0/1 case 1（precision_failed） | A1: 20/20<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 153,865 / 62,199 / 3,112,448 / **3,328,512** | 2 + 2 = **4** |
| Mish | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 123,312 / 61,594 / 5,015,808 / **5,200,714** | 1 + 1 = **2** |
| MoeFinalizeRouting | 0/1 case 1（precision_failed） | A1: runtime_error（无 full 计数）<br>A2: 20/20<br>A3: 20/20<br>A4: 20/20 | 20/20 + 74/80 | 平台阻塞 | 4（4） | 353,239 / 162,289 / 10,255,872 / **10,771,400** | 2 + 2 = **4** |
| MoeGatingTopKSoftmax | 0/1 case 1（precision_failed） | A1: 未形成候选（未验证）<br>A2: provider_error（未验证）<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 483,489 / 236,120 / 27,610,368 / **28,329,977** | 1 + 1 = **2** |
| MoeReRouting | 0/1 case 1（precision_failed） | A1: 0/20（precision_failed）<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 170,366 / 74,429 / 3,721,984 / **3,966,779** | 1 + 1 = **2** |
| NMS | 0/1 case 1（precision_failed） | A1: source_integrity_gate（未执行精度）<br>A2: 20/20<br>A3: 20/20<br>A4: 20/20 | 20/20 + 80/80 | 严格通过 | 4（4） | 581,628 / 310,318 / 25,606,144 / **26,498,090** | 3 + 3 = **6** |
| QuantMatmul | 0/1 case 1（precision_failed） | A1: precision_failed（无 full 计数）<br>A2: cpp_regression_gate（未执行精度）<br>A3: 20/20<br>A4: provider_error（未验证）<br>A5: execution_aborted（无 full 计数）<br>A6: 26/26<br>A7: 26/26 | 20/20 + 80/80 | 严格通过 | 7（6） | 929,105 / 546,176 / 40,752,896 / **42,228,177** | 4 + 3 = **7** |
| RmsNorm | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: 20/20<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 240,910 / 108,121 / 6,029,056 / **6,378,087** | 2 + 2 = **4** |
| ROIAlign | 0/1 case 1（precision_failed） | A1: provider_error（未验证）<br>A2: 20/20<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 538,094 / 335,974 / 36,042,604 / **36,916,672** | 2 + 2 = **4** |
| Scatter | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 运行中 | 隐藏评测中 | 1（1） | 316,388 / 196,323 / 29,530,368 / **30,043,079** | 1 + 1 = **2** |
| Sigmoid | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 83,284 / 40,477 / 3,164,928 / **3,288,689** | 1 + 1 = **2** |
| Softmax | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 136,886 / 76,076 / 4,688,640 / **4,901,602** | 1 + 1 = **2** |
| StridedSlice | 0/1 case 1（precision_failed） | A1: 20/20<br>A2: 20/20 | 20/20 + 80/80 | 严格通过 | 2（2） | 150,486 / 77,492 / 4,662,784 / **4,890,762** | 2 + 2 = **4** |
| SwiGlu | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 116,731 / 84,434 / 5,002,496 / **5,203,661** | 1 + 1 = **2** |
| TopK | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 211,980 / 134,054 / 12,954,368 / **13,300,402** | 1 + 1 = **2** |
| Transpose | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 130,125 / 100,011 / 6,001,920 / **6,232,056** | 1 + 1 = **2** |
| Unique | 0/1 case 1（runtime_error） | A1: 20/20<br>A2: 20/20<br>A3: 20/20 | 20/20 + 80/80 | 严格通过 | 3（3） | 500,318 / 322,882 / 28,381,184 / **29,204,384** | 3 + 3 = **6** |
| UnsortedSegmentSum | 0/1 case 1（precision_failed） | A1: 20/20 | 20/20 + 80/80 | 严格通过 | 1（1） | 96,950 / 54,360 / 3,632,384 / **3,783,694** | 1 + 1 = **2** |
| WeightQuantBatchMatmul | 0/1 case 1（precision_failed） | A1: cpp_regression_gate（未执行精度）<br>A2: 20/20<br>A3: 20/20<br>A4: provider_error（未验证）<br>A5: 20/20<br>A6: provider_error（未验证）<br>A7: provider_error（未验证）<br>A8: provider_error（未验证）<br>A9: 20/20<br>A10: provider_error（未验证） | 15/20 + 未运行 | 待定 | 10（10） | 1,251,789 / 738,400 / 43,233,792 / **45,223,981** | 4 + 0 = **4** |

### 13. 汇总与观察

| 汇总项 | 数值 |
|---|---:|
| 已提交算子 | 45 |
| 严格通过 | 36 |
| 待定 / 未严格通过 / 平台阻塞 / 隐藏在途 | 4 / 2 / 2 / 1 |
| Debug-Agent 实际模型会话 | 148 |
| 其中报告非零 token usage 的会话 | 144 |
| attempt 后有权威验证记录 | 110 |
| 其中完成本地官方 full-eval | 98 |
| 未进入验证 | 38（provider 失败 36，未形成候选 2） |
| A1 即完成本地 full 全通过的算子 | 21/45 |
| 输入 token | 19,968,820 |
| 输出 token | 10,998,185 |
| cache-read token | 988,189,842 |
| 总处理 token | 1,019,156,847 |
| CANNBench 标准 job | 83 |
| CANNBench 隐藏 job | 65 |
| CANNBench job 总数 | 148 |

资源投入最高的五个算子是：

| 算子 | Agent attempts | 总处理 token | 官网 jobs | 快照状态 |
|---|---:|---:|---:|---|
| GroupedMatmul | 17 | 187,107,620 | 14 | 待定 |
| LSTM | 6 | 120,411,663 | 6 | 待定 |
| ArgMax | 4 | 50,555,963 | 2 | 严格通过 |
| DepthwiseConv2D | 9 | 46,423,426 | 4 | 严格通过 |
| WeightQuantBatchMatmul | 10 | 45,223,981 | 4 | 待定 |

这组结果说明：

- 98 次完整本地 full-eval 中，83 次为 `20/20`、2 次为历史快照的 `26/26`；其余为 `0/20` 2 次、`2/20` 1 次、`18/20` 6 次、`19/20` 4 次。另有 1 次 build failure、7 次验证前门禁中止、4 次已运行验证但无 full 逐 case 计数。
- 21/45 个算子在 A1 就达到本地 full 全通过。这只能说明第一次真实语义候选已经通过公开集，不能推导它也能一次通过隐藏集。
- `ForeachAddcdivScalar`（`0/20 runtime_error -> 20/20`）、`MaskedScale`（`19/20 -> 20/20`）和 `Conv2D`（`2/20 -> 20/20`）展示了 Debug-Agent 迭代直接改善本地公开精度的过程。
- `EngramGateFusion` 三次本地均为 `20/20`，榜单隐藏却只有 `9/80`；`GroupedMatmul` 多次本地 `20/20`，隐藏为 `55/80`；LSTM 多次本地 `20/20`，隐藏为 `61/80`。这些轨迹说明继续重复公开集成功并不能解决隐藏分布覆盖问题。
- `WeightQuantBatchMatmul` 多次本地 `20/20`，官网公开快照却是 `15/20`。这属于本地与官网结果不一致的事实，但仅凭现有证据不能把原因唯一归结为 910B/910C 设备差异；case/evaluator/version 与运行环境差异也仍需排查。
- 隐藏失败 continuation 会显著抬高 Agent 和官网成本，但两者不是线性关系。例如 `GroupedMatmul` 有 17 次 Agent 调用、14 个官网 job，而 `WeightQuantBatchMatmul` 有 10 次 Agent 调用、4 个标准 job；大量本地迭代不会自动转化为官网提交。
- “官网 job 数”与“三次失败转待定”的 policy cycle 不是同一指标。前者统计整个归档中的实际服务端 job；后者只统计满足当时不可变候选、完整门禁和失败分类规则的连续 cycle。
- cache-read 占总处理 token 的 96.96%，所以 10.19 亿总处理 token 不能理解为 10.19 亿新输入 token；新输入加输出合计为 30,967,005。

### 14. 可复现证据

- 结构化明细：[`operator_effort_snapshot_20260812.json`](./operator_effort_snapshot_20260812.json)
- 统计脚本：[`analyze_operator_effort_snapshot.py`](./analyze_operator_effort_snapshot.py)
- 明细 JSON SHA-256：`d1fc03be0ab153abe3d7e4de07282a90351671fc53bf7ab60dbf4d8663546631`
- 统计脚本 SHA-256：`fda58f5b2630474c384be2fb846b6d61e38ef3eb2d561a67918a35a4507bf268`
- JSON 以 operator、session ID 和 job ID 为主键保存初始证据路径、token 分项和标准/隐藏 job ID，便于逐项复核。
- 脚本从远端只读扫描 `/home/wsx/AscendOpGenAgent/outputs`，按 `--snapshot-at 2026-08-12T01:01:00+08:00` 截断；不会修改任务、候选或官网状态。

## 第一名 artifact 对比与通用 AscendC 优化 KB（2026-08-13）

本 section 是独立于前文 2026-08-12 排行榜报告的新快照，不修改前文的排名、精度、attempt 或 token 统计。

### 15. 目标与结论

本轮使用第一名 `CANNBOT LINGXI-EVO` 的聚合导出包，对其逐算子 TOP1 artifact 与我方实时榜单结果进行比较。筛选规则为：第一名阶段全部 case 通过、加速比严格大于 `1.0x`，并且严格高于我方同算子同阶段。

共得到 **39 条 operator/stage 记录，涉及 26 个算子**。其中 22 条记录来自同一不可变 artifact 的隐藏全通过结果，覆盖 17 个算子，可作为通用知识提炼候选；公开集领先但缺少同 artifact 隐藏全通过证据的记录只保留为假设；隐藏正确性不完整的 artifact 只作为反例。

从通过隐藏全量验证的源码中提炼出 **13 条硬件无关的通用 AscendC 优化知识**，其中 12 条为 `promoted`、1 条为 `supported`。当前只提供只读检索和 shadow 门禁，不会自动注入现有精度 Debug-Agent。

### 16. 逐阶段性能对比

| 算子 | 阶段 | 第一名 | 我方 | 差值 | 第一名正确性 | 证据处置 |
|---|---|---:|---:|---:|---:|---|
| Exp | 公开 | 2.113x | 1.871x | +0.242x | 20/20 | 仅公开证据 |
| ForeachNorm | 公开 | 1.569x | 0.909x | +0.660x | 20/20 | 仅公开证据 |
| MaskedScale | 公开 | 2.456x | 2.236x | +0.220x | 20/20 | 仅公开证据 |
| Mish | 公开 | 1.885x | 0.939x | +0.946x | 20/20 | canonical 候选 |
| SwiGlu | 公开 | 1.547x | 1.410x | +0.137x | 20/20 | 仅公开证据 |
| ApplyAdamW | 公开 | 1.860x | 1.358x | +0.502x | 20/20 | 仅公开证据 |
| ApplyRotaryPosEmb | 公开 | 1.086x | 0.127x | +0.959x | 20/20 | 仅公开证据 |
| ArgMax | 公开 | 1.342x | 0.295x | +1.047x | 20/20 | 仅公开证据 |
| DynamicQuant | 公开 | 1.368x | 0.558x | +0.810x | 20/20 | canonical 候选 |
| Gather | 公开 | 1.074x | 0.031x | +1.043x | 20/20 | 仅公开证据 |
| Gcd | 公开 | 1.001x | 0.102x | +0.899x | 20/20 | 同 artifact 隐藏未全过 |
| GroupNorm | 公开 | 1.022x | 0.402x | +0.620x | 20/20 | canonical 候选 |
| ResizeBilinear | 公开 | 2.023x | 0.060x | +1.963x | 20/20 | 仅公开证据 |
| RmsNorm | 公开 | 1.054x | 0.503x | +0.551x | 20/20 | canonical 候选 |
| Softmax | 公开 | 1.003x | 0.424x | +0.579x | 20/20 | 仅公开证据 |
| UnsortedSegmentSum | 公开 | 1.215x | 0.141x | +1.074x | 20/20 | canonical 候选 |
| AddRmsNormDynamicQuant | 公开 | 1.153x | 0.352x | +0.801x | 20/20 | 仅公开证据 |
| Conv3DBackpropFilter | 公开 | 1.070x | 0.001x | +1.069x | 20/20 | 同 artifact 隐藏未全过 |
| DequantSwigluQuant | 公开 | 1.248x | 0.491x | +0.757x | 20/20 | 仅公开证据 |
| MhcSinkhorn | 公开 | 1.019x | 0.158x | +0.861x | 20/20 | 同 artifact 隐藏未全过 |
| NMS | 公开 | 1.286x | 0.038x | +1.248x | 20/20 | 同 artifact 隐藏未全过 |
| StridedSlice | 公开 | 1.906x | 0.168x | +1.738x | 20/20 | 同 artifact 隐藏未全过 |
| ApplyAdamW | 隐藏 | 4.045x | 3.190x | +0.855x | 80/80 | canonical 候选 |
| ApplyRotaryPosEmb | 隐藏 | 1.099x | 0.365x | +0.734x | 80/80 | canonical 候选 |
| ArgMax | 隐藏 | 1.167x | 0.331x | +0.836x | 80/80 | canonical 候选 |
| DynamicQuant | 隐藏 | 3.846x | 1.795x | +2.051x | 80/80 | canonical 候选 |
| Exp | 隐藏 | 3.654x | 3.302x | +0.352x | 80/80 | canonical 候选 |
| ForeachNorm | 隐藏 | 3.389x | 2.143x | +1.246x | 80/80 | canonical 候选 |
| Gather | 隐藏 | 1.780x | 0.246x | +1.534x | 80/80 | canonical 候选 |
| Gelu | 隐藏 | 2.378x | 1.970x | +0.408x | 80/80 | canonical 候选 |
| GroupNorm | 隐藏 | 2.211x | 1.284x | +0.927x | 80/80 | canonical 候选 |
| Mish | 隐藏 | 3.120x | 1.894x | +1.226x | 80/80 | canonical 候选 |
| QuantMatmul | 隐藏 | 1.513x | 0.303x | +1.210x | 80/80 | canonical 候选 |
| RmsNorm | 隐藏 | 1.929x | 1.276x | +0.653x | 80/80 | canonical 候选 |
| Sigmoid | 隐藏 | 3.888x | 3.232x | +0.656x | 80/80 | canonical 候选 |
| Softmax | 隐藏 | 1.494x | 1.239x | +0.255x | 80/80 | canonical 候选 |
| SwiGlu | 隐藏 | 3.012x | 2.836x | +0.176x | 80/80 | canonical 候选 |
| Transpose | 隐藏 | 1.153x | 0.277x | +0.876x | 80/80 | canonical 候选 |
| UnsortedSegmentSum | 隐藏 | 1.200x | 0.453x | +0.747x | 80/80 | canonical 候选 |

“canonical 候选”表示其正确性与性能证据允许参与抽象，不表示源码中的每个技巧都已通过单变量因果实验。最终 KB 只晋级有明确机制、风险边界和复现要求的知识。

### 17. 第一名导出包不是单一实现

第一名导出包包含 83 个 submission ZIP；排行榜的每个 operator/stage 可以来自不同 submission 和不同用户。因此，本轮没有把整个压缩包视为一个统一实现，而是逐行读取 `source.submission_id`，再打开对应内层 ZIP，并记录 inner ZIP SHA、源码文件 SHA、job 和行号。

这也解释了为什么公开和隐藏阶段可能来自不同 artifact。公开领先不能自动证明同一源码隐藏也正确；只有同一不可变 artifact 的隐藏全通过记录才能进入 canonical 提炼路径。

### 18. 我方源码关联的更正

我方 53 个算子任务均有实现。早期“18/26 可做源码对照”是索引覆盖不足，不是 8 个算子没有源码：旧索引只扫描 task root，没有完整纳入 `website_submissions`、不可变 ZIP 和 candidate inventory。

修复索引后：

- 26 个筛选算子中，25 个至少有榜单 job 对应的精确源码 SHA；
- 24 个三文件或更多提交源码完整；
- `MhcSinkhorn` 为部分精确关联，kernel 和 launch 已匹配，但提交时 plugin 的相同 SHA 未同步到本地；
- `QuantMatmul` 有实现和多份历史 ZIP，但缺少把当前榜单 `job_cf45c1fee2cf` / `job_8a8dd50c921b` 绑定到其中某一不可变版本的记录，所以不能猜测版本。

其余 27 个算子没有进入源码对照，是因为不满足本轮性能筛选条件，而不是缺少实现。

### 19. 提炼出的通用优化知识

| 类型 | 通用机制 | 状态 |
|---|---|---|
| `OPT_UB_RESIDENT_FAST_PATH` | UB 可驻留形状走单遍快路径，大形状走流式回退 | promoted |
| `OPT_ROW_BATCHING` | 合批多行或多 group，摊薄标量同步和小 DMA | promoted |
| `OPT_PIPELINE_OVERLAP` | 双缓冲及精确事件重叠搬运、计算、写回 | promoted |
| `OPT_SPECIALIZED_PATH_ROUTING` | 依据公开 shape/dtype/layout/属性选择少量专用路径 | promoted |
| `OPT_VECTOR_PASS_ELIMINATION` | 消除无操作向量趟并融合系数、中间量和写回 | promoted |
| `OPT_ALGEBRAIC_REFORMULATION` | 用等价且代价更低的数学形式替换昂贵复合原语 | promoted |
| `OPT_HIERARCHICAL_REDUCTION` | 先局部聚合，再做少量全局原子或最终归约 | promoted |
| `OPT_IRREGULAR_MEMORY_ACCESS` | 不规则访存使用并行 load、Gather、预取和窄写打包 | supported |
| `OPT_UB_BUFFER_REUSE` | 按生命周期复用 UB，精确预算后放大 tile | promoted |
| `OPT_ALIGNED_MAIN_LOOP` | 对齐主循环走大块拷贝，尾块单独处理 | promoted |
| `OPT_CORE_LOAD_BALANCE` | 按真实工作量均衡分核 | promoted |
| `OPT_DTYPE_AWARE_FAST_MATH` | 按输出 dtype 误差预算选择 fast math 与精确回退 | promoted |
| `OPT_MEASURED_ROLLBACK_POLICY` | 把负向性能实验作为证据，按 case 回退 | promoted |

canonical KB 不包含 `hardware_scope`，不按 910B/910C 检索或限制适用范围。根据 owner 提供的信息，第一名团队和我方均在 910B 上开发，再提交至 CANNBench 910C 评测；这进一步说明不应把知识描述为“910C 专属经验”。榜单硬件只保留在审计快照中，用于解释测量来源。

性能候选验收时仍需冻结环境哈希，这是为了让 baseline 与 candidate 的重复计时可比较，不是把知识绑定到硬件型号。

### 20. Debug-Agent 集成设计

采用“先精度，后性能”的串联流程：

```text
precision Debug-Agent
  -> official full / anti-cheat / target compile 全通过
  -> 冻结正确 baseline 与重复计时
  -> optimization coarse retrieval
  -> profiling 与 bottleneck 分类
  -> optimization fine retrieval
  -> 单一可归因修改
  -> 先复跑完整正确性
  -> 再比较同 case 重复计时
  -> Pareto accept 或 rollback
```

不能先优化错误实现，也不能在同一个 attempt 同时做大范围正确性修复和性能改造。否则“更快”可能来自漏算，且无法归因。

当前 Phase 1 已完成：证据审计、13 条硬件无关 KB、只读 coarse/fine 检索器和 entry/Pareto 纯函数门禁。尚未自动接入 runner。后续还需要统一重复计时 adapter、opt-in 单轮性能状态机和受控消融；只有它们证明收益稳定且正确性不退化，才考虑默认开启。

### 21. 文件与哈希

- 完整设计与实施记录：[`ascendc_optimization_kb_and_debug_agent_integration_20260813.md`](../../docs/ascendc_optimization_kb_and_debug_agent_integration_20260813.md)
- canonical KB：[`optimization_knowledge_base.json`](../../skills/ascendc/ascendc-debug/references/optimization_knowledge_base.json)
- 配对 JSON：[`paired_operator_stage_metrics.json`](./optimization_knowledge_extraction_20260813/paired_operator_stage_metrics.json)
- 配对 CSV：[`paired_operator_stage_metrics.csv`](./optimization_knowledge_extraction_20260813/paired_operator_stage_metrics.csv)
- KB manifest：[`optimization_kb_manifest.json`](./optimization_knowledge_extraction_20260813/optimization_kb_manifest.json)
- 非 canonical 假设与反例：[`optimization_hypotheses_and_counterexamples.json`](./optimization_knowledge_extraction_20260813/optimization_hypotheses_and_counterexamples.json)
- canonical KB SHA-256：`a5f2586572170827ccdf2675e89629b02b0e552a508b643d2719d55d618748bf`
- 配对 JSON SHA-256：`2a5c923486a3fdeee5478e3e890fe4b4a525e3a785a0555a477017670f8818f4`
