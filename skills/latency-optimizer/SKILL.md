---
name: latency-optimizer
description: >
  算子性能优化 Skill — 分析现有 Triton 算子实现，识别性能瓶颈，
  应用优化策略，并进行自动调优以达到目标加速比。
argument-hint: >
  输入：code-file-path、op-name、target-speedup（默认1.5x）、warmup、repeats。
  输出：优化后的代码、性能数据、是否达到目标加速比。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# Latency Optimizer Skill

<role>
你是一个算子性能优化专家。你的任务是分析现有 Triton 算子实现，识别性能瓶颈，
应用优化策略，并通过自动调优使算子达到目标加速比。
</role>

## 优化流程

```
输入：code_file + target_speedup
    ↓
[1. 代码分析] → 识别瓶颈点
    ↓
[2. 优化策略生成] → 制定优化方案
    ↓
[3. 代码重写] → 应用优化
    ↓
[4. 验证] → 确保正确性
    ↓
[5. 自动调优] → 搜索最优参数
    ↓
[6. 性能评估] → 对比目标加速比
    ↓
输出：优化后的代码 + 性能数据 + 是否达标
```

---

## TODO: Step 1 - 代码分析

识别以下性能瓶颈：
- 内存访问模式（ coalesced / strided / random ）
- 计算密度（ arithmetic intensity ）
- 指令级并行度
- 线程块配置合理性
- Shared Memory 使用情况
- 自动调优潜力

---

## TODO: Step 2 - 优化策略生成

基于分析结果生成优化策略：
- 向量化加载/存储
- 阻塞优化（ blocking ）
- 共享内存复用
- 指令重排
- 自动调优参数选择

---

## TODO: Step 3 - 代码重写

应用优化策略重写算子代码。

---

## TODO: Step 4 - 验证

调用 kernel-verifier 的验证脚本确保正确性。

---

## TODO: Step 5 - 自动调优

使用 Triton 的 auto-tune 机制或参数搜索优化性能。

---

## TODO: Step 6 - 性能评估

对比优化前后的性能数据，判断是否达到目标加速比。

---

## 脚本位置

性能测试脚本位于 `kernel-verifier` skill 的 `scripts/benchmark.py`。

---

## 依赖

- kernel-verifier skill（用于验证和性能测试）
- Triton Ascend 相关工具链
