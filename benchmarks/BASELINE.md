# Baseline

> main @ latest, 2026-03-28
> Device: Ascend 910B2, CANN 8.0, PyTorch 2.1
> 评测范围: Level 1 (34 tasks), Level 2 (6 tasks), 共 40 tasks

## KernelBench

### 评测子集列表：

Level 1：2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 57, 61, 63, 64, 67, 82, 87, 99, 100

Level 2：6, 12, 17, 23, 30, 94
### 基线结果（更新于2026-03-28）

| Level | Problem ID | Problem | 编译 | 精度 | PyTorch (ms) | Generated (ms) | Speedup |
|:---:|:---:|---------|:---:|:---:|---:|---:|---:|
| 1 | 48 | Mean_reduction_over_a_dimension | Pass | Pass | 7.69 | 716.79 | 0.01x |
| 1 | 50 | conv_standard_2D_square_input_square_kernel | Pass | Pass | - | - | - |
| 1 | 51 | Argmax_over_a_dimension | Pass | Pass | - | - | - |
| 1 | 53 | Min_reduction_over_a_dimension | Pass | Pass | 18.13 | 22.97 | 0.79x |
| 1 | 54 | conv_standard_3D_square_input_square_kernel | Pass | Pass | - | - | - |
| 1 | 57 | conv_transposed_2D_square_input_square_kernel | Pass | Pass | - | - | 1.00x |
| 1 | 61 | conv_transposed_3D_square_input_square_kernel | Pass | Pass | - | - | 0.86x |
| 1 | 63 | conv_standard_2D_square_input_square_kernel | Pass | Pass | 23.59 | 166.72 | 0.14x |
| 1 | 64 | conv_transposed_1D | Pass | Pass | - | - | 1.00x |
| 1 | 67 | conv_standard_1D | Pass | Pass | - | - | - |
| 1 | 82 | conv_depthwise_2D_square_input_square_kernel | Pass | Pass | - | - | 0.12x |
| 1 | 87 | conv_pointwise_2D | Pass | Pass | 31.63 | 4304.04 | 0.01x |
| 1 | 99 | TripletMarginLoss | Pass | Pass | 10.68 | 4.60 | 2.32x |
| 1 | 100 | HingeLoss | Pass | Pass | 31.67 | 1283.66 | 0.02x |
| 2 | 6 | Conv3d_Softmax_MaxPool_MaxPool | Pass | Pass | 0.48 | - | 0.00x |
| 2 | 12 | Gemm_Multiply_LeakyReLU | Pass | Pass | 0.61 | 0.62 | 0.98x |
| 2 | 17 | Conv2d_InstanceNorm_Divide | Pass | Pass | 3.03 | 3.25 | 0.93x |
| 2 | 23 | Conv3d_GroupNorm_Mean | Pass | Pass | 0.65 | 0.65 | 1.00x |
| 2 | 30 | Gemm_GroupNorm_Hardtanh | Pass | Pass | 0.66 | 0.63 | 1.04x |
| 2 | 94 | Gemm_BiasAdd_Hardtanh_Mish_GroupNorm | Pass | Pass | 0.66 | 0.66 | 1.00x |

编译通过率：20/20
精度通过率：20/20
综合通过率（编译+精度均过）：20/20
平均 Speedup：0.64x（仅统计有性能数据的 16 个任务）

（TODO）
以下 Level 1 任务待补充评测结果：
2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46
