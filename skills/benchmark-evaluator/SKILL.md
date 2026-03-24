---
name: benchmark-evaluator
description: >
  Benchmark Evaluator Skill — 串行执行算子评测任务，调用指定 Agent 生成代码并验证。
  接收已解析的参数，返回每个任务的结构化结果。
argument-hint: >
  必需：agent_name, agent_workspace, level_problems, benchmark_path (绝对路径), arch, npu_id, output_path (绝对路径)。
  可选：timeout_per_task, warmup, repeats, completed_tasks (用于断点续跑)。
---

# Benchmark Evaluator Skill

<role>
你是一个自动化评测任务执行器。你的任务是串行执行 KernelBench 评测任务，调用指定的 Agent 生成代码，验证正确性，测试性能，并返回每个任务的结构化结果。
</role>

---

## 📥 输入参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 | 由谁提供 |
|------|------|------|------|---------|
| `agent_name` | str | 被评测的 Agent 名称 | `"triton-ascend"` | Agent |
| `agent_workspace` | str | Agent 工作区路径 | `"/root/.opencode"` | Agent |
| `benchmark_path` | str | **已解析的绝对路径** | `"/root/.opencode/benchmarks/KernelBench"` | **Agent 解析后传入** |
| `level_problems` | dict | 评测范围 | `{1: [1,2], 2: null}` | Agent |
| `arch` | str | 硬件架构 | `"ascend910b2"` | **Agent 检测后传入** |
| `npu_id` | int | NPU 设备 ID | `0` | **Agent 选择后传入** |
| `output_path` | str | **根输出目录的绝对路径** | `"/root/.opencode/benchmark_results/triton-ascend_20250324_103000_1234"` | **Agent 创建并传入** |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timeout_per_task` | int | 2400 | 单任务超时（秒）|
| `warmup` | int | 5 | 性能测试 warmup 次数 |
| `repeats` | int | 50 | 性能测试重复次数 |
| `completed_tasks` | list | `[]` | 已完成任务列表（用于断点续跑）|

### completed_tasks 格式

```json
[
  {"level": 1, "problem_id": 1},
  {"level": 1, "problem_id": 2},
  {"level": 2, "problem_id": 1}
]
```

---

## 🔄 工作流程

```
Phase 1: 初始化
  ├── 验证输入参数完整性
  ├── 验证 benchmark_path 存在且有效
  ├── 设置环境变量 ASCEND_RT_VISIBLE_DEVICES={npu_id}
  └── 创建输出目录结构

Phase 2: 任务扫描
  ├── 根据 level_problems 扫描 benchmark_path
  ├── 构建任务列表 [(level, problem_id, task_file)]
  ├── 根据 completed_tasks 过滤已完成任务
  └── 确定待执行任务队列

Phase 3: 串行执行
  └── 对于每个任务：
      ├── 调用 kernelgen-workflow 生成代码
      ├── 正确性验证（kernel-verifier skill）
      ├── 性能测试（benchmark.py）
      ├── 保存结果到 verify_result.json 和 perf_result.json
      └── **返回任务结果给 Agent**

Phase 4: 完成
  └── 返回执行摘要
```

---

## 📤 返回结果格式

### 单个任务结果

```json
{
  "level": 1,
  "problem_id": 1,
  "op_name": "matmul",
  "status": "success|failed|timeout",
  "error_type": null|"compilation"|"verification"|"performance",
  "error_message": "具体错误信息",
  "verify_result": {
    "passed": true,
    "max_diff": 0.001,
    "details": "..."
  },
  "perf_result": {
    "speedup": 1.5,
    "triton_time_ms": 10.5,
    "pytorch_time_ms": 15.8
  },
  "output_path": "<output_path>/level_1/1_matmul/",
  "execution_time_seconds": 120.5
}
```

### 最终执行摘要

```json
{
  "total_tasks": 100,
  "completed_tasks": 95,
  "failed_tasks": 5,
  "timeout_tasks": 0,
  "total_execution_time_seconds": 12000,
  "results": [
    {/* 单个任务结果（包含 PyTorch vs 生成代码的性能对比）*/},
    {/* 单个任务结果（包含 PyTorch vs 生成代码的性能对比）*/},
    ...
  ]
}
```

---

## 🎯 核心职责

### 1. 任务扫描

- 根据 `level_problems` 扫描 `benchmark_path` 目录
- 解析每个任务文件的元数据
- 根据 `completed_tasks` 过滤已完成任务
- 构建待执行任务队列

### 2. 代码生成

**直接调用 kernelgen-workflow**：

```bash
opencode run --agent kernelgen-workflow "生成并验证算子代码..."
```

### 3. 正确性验证

- 调用 `kernel-verifier` skill
- 对比生成代码与 PyTorch 参考实现的输出
- 记录最大差异值（Max Diff）

### 4. 性能测试

- 调用 `benchmark.py` 脚本
- 执行 warmup 和 repeats 次测试
- 计算加速比（Triton vs PyTorch）

### 5. 结果返回

- **每完成一个任务，立即返回结果给 Agent**
- 不生成报告（报告由 Agent 负责）
- 不维护状态文件（状态由 Agent 维护）

---

## 📁 输出目录结构

Skill 在传入的 `output_path` 目录下创建任务子目录：

```
{output_path}/                                      ← 由 Agent 创建并传入
├── level_{n}/                                      ← Skill 创建
│   └── {problem_id}_{op_name}/                     ← Skill 创建
│       ├── generated_code.py                       ← Skill 保存
│       ├── verify_result.json                      ← Skill 保存
│       └── perf_result.json                        ← Skill 保存
└── ...
```

**注意**：
- `output_path` 是**完整的根目录绝对路径**，由 Agent 创建
- Skill **不添加** `run_{timestamp}/agent_{agent_name}/` 等中间层级
- Skill **不创建** `agent_report.md`（由 Agent 维护）
- Skill **不维护** `.benchmark_state.json`（由 Agent 维护）

---

## 💡 使用示例

### 示例 1: 基础调用

```python
{
  "agent_name": "triton-ascend",
  "agent_workspace": "/root/.opencode",
  "benchmark_path": "/root/.opencode/benchmarks/KernelBench",  # 已解析的绝对路径
  "output_path": "/root/.opencode/benchmark_results/triton-ascend_20250324_103000_1234",  # Agent 创建的根目录绝对路径
  "level_problems": {1: [1, 2, 3]},
  "arch": "ascend910b2",  # Agent 检测后传入
  "npu_id": 0  # Agent 选择后传入
}
```

### 示例 2: 断点续跑

```python
{
  "agent_name": "triton-ascend",
  "agent_workspace": "/root/.opencode",
  "benchmark_path": "/root/.opencode/benchmarks/KernelBench",
  "output_path": "/root/.opencode/benchmark_results/triton-ascend_20250324_103000_1234",  # Agent 创建的根目录
  "level_problems": {1: [1, 2, 3, 4, 5]},
  "arch": "ascend910b2",
  "npu_id": 0,
  "completed_tasks": [  # Agent 从状态文件加载后传入
    {"level": 1, "problem_id": 1},
    {"level": 1, "problem_id": 2}
  ]
}
```

---

## 注意事项

1. **参数预处理**：
   - `benchmark_path` 必须是**已解析的绝对路径**
   - `arch` 和 `npu_id` 由 Agent 检测/选择后传入
   - Skill 不负责参数收集和解析

2. **断点续跑**：
   - Skill 根据 `completed_tasks` 跳过已完成任务
   - 状态文件由 Agent 维护，Skill 不操作

3. **结果返回**：
   - 每完成一个任务，立即返回结果给 Agent
   - 不生成报告，不维护状态文件

4. **错误处理**：
   - 单任务失败不影响整体流程
   - 记录错误信息并返回给 Agent
   - 超时任务标记为 `timeout` 状态

5. **串行执行**：
   - 任务按顺序逐个执行，不进行并行化
   - 保证资源独占，避免 NPU 冲突

6. **环境隔离**：
   - 设置 `ASCEND_RT_VISIBLE_DEVICES={npu_id}` 确保 NPU 独占
   - 每个任务独立的输出目录

---

## 依赖

- Python 3.8+
- opencode Agent 调用机制
- kernelgen-workflow subagent
- kernel-verifier skill
- KernelBench 数据集
- NPU 设备（用于验证和性能测试）
