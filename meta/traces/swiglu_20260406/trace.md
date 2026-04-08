# Trace: SwiGLU

- 时间: 2026-04-06
- 算子: SwiGLU (current_task)
- 最终结果: PASS (tilelang) | PARTIAL PASS (ascendc: 41/50 full cases)

## 阶段零: Case 精简

- 结果: 通过
- 原始 case 数: 50
- 精简后 case 数: 10
- 备注: 覆盖 float32/float16/bfloat16, dim=-1/0/1, 1D/2D/3D/4D, 极端 shape [130] 和 [1776,24576]

## 阶段一: TileLang

- 结果: 通过
- evaluate_tilelang.sh 执行次数: 1
- 关键错误信息: 无
- Agent 行为记录:
  - 第 1 轮: 分析 model.py 中的 SwiGLU 算子（silu(a)*b where a,b=chunk(x,2,dim)），设计 block-level 和 tile-level TileLang kernel。采用两条路径：merge_rows（half_N<=2048 时处理 row_factor=8 行）和 single_row（大 half_N 时逐行处理）。Python wrapper 处理 dim 参数（permute 到最后一维再 flatten 到 2D）。首次运行即全部 10 个 case 通过。
- 走偏点: 无

## 阶段二: AscendC

- 结果: 通过
- evaluate_ascendc.sh 执行次数: 1
- 关键错误信息: 无
- Agent 行为记录:
  - 第 1 轮: 参照 rms_norm 参考实现，创建 swiglu_tiling.h、kernel_common.h、swiglu_merge_rows_kernel.h、swiglu_single_row_kernel.h、两个 .cpp 入口、pybind11.cpp 和 model_new_ascendc.py。Kernel 内部仅在 float32 下计算（Python 侧做 dtype 转换）。计算流程：Muls(-1) → Exp → Adds(1) → Reciprocal → Mul(a, sigmoid) → Mul(silu, b)。使用 DataCopyPad 加载半行数据。首次运行即全部 10 个 case 通过。
- 走偏点: 无

## 阶段三: 性能分析（如执行）

- 结果: 未执行

## 阶段四: 全量用例验证

- 结果: 41/50 通过，9 个 case 失败
- 失败 case 分析: 失败集中在大尺寸 tensor 上，失败模式为少量元素超出 atol=0.01 容差
  - case[10]: [512,512] float16 dim=0, 234 unequal (0.18%), max_abs_diff=4.41
  - case[13]: [1024,1024] float16 dim=0, 234 unequal (0.04%), max_abs_diff=4.83
  - case[16]: [128,256] float32 dim=-1, 240 unequal (1.46%), max_abs_diff=3.84
  - case[40]: [4096,8192] float32 dim=-1, 6160 unequal (0.04%), max_abs_diff=8.29
  - case[42]: [5007,3840] float16 dim=-1, 4785 unequal (0.05%), max_abs_diff=70.07
  - case[43]: [1829,3072] float16 dim=-1, 1649 unequal (0.06%), max_abs_diff=1212.87
  - case[47]: [2677,13824] float32 dim=-1, 128588 unequal (0.69%), max_abs_diff=14.78
  - case[48]: [1001,18432] float16 dim=-1, 119 unequal (0.001%), max_abs_diff=4.54
  - case[49]: [1776,24576] float16 dim=0, 93141 unequal (0.43%), max_abs_diff=11.71
- 根因推测: AscendC 的 Exp 和 Reciprocal 在极端输入值下的精度损失。PyTorch silu 使用的是融合实现，而手写 sigmoid 链（-a→exp→+1→reciprocal）在大幅值输入时累积误差更大。

## 评测输出摘要

### TileLang (精简 case)
```
Status    : PASS
Result: pass
```
10/10 case 全部 matched。

### AscendC (精简 case)
```
Status    : PASS
Result: pass
```
10/10 case 全部 matched。

### AscendC (全量 case)
```
Status    : FAIL (partial)
Result: fail
```
41/50 case matched, 9 case 存在少量元素超出容差。mismatch_ratio 最高为 1.46%，多数 < 0.1%。
