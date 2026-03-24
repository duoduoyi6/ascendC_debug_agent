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

## 参考资料索引

| 参考文档 | 用途 |
|---------|------|
| `vector_cmp.md` | 向量比较方法 |

