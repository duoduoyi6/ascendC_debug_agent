---
name: ascend-benchmark-evaluator
description: Conduct a comprehensive evaluation of the Lingxi-Code Agent’s ability to generate Ascend C code using datasets such as NPUKernelbench. 
---
## What I do

针对NPUKernelbench等数据集，对Lingxi-Code Agent生成Ascend C代码的能力进行完整评估。支持批量算子测试、正确性验证、性能对比和综合报告生成。

## When to use me

当你需要：
- 评估Agent在大量算子上的代码生成能力
- 对NPUKernelbench数据集进行自动化测试
- 生成算子实现的benchmark报告
- 对比参考实现和自定义实现的正确性与性能

## Workflow

### 1. 数据集结构准备

支持两种数据集格式：

**NPUKernelBench标准格式（推荐）**：
```
NPUKernelBench/level1/
├── 1_GELU.py                     # PyTorch参考实现
├── 1_GELU.json                   # 测试用例（JSON Lines格式）
├── 2_SwiGLU.py
├── 2_SwiGLU.json
├── 3_Add.py
└── 3_Add.json
```

文件命名规范：`{id}_{OpName}.{ext}`
- `id`: 数字编号
- `OpName`: 算子名称（如 GELU, SwiGLU, Add）
- `.py`: 参考实现文件（包含 Model 类、get_inputs()、get_init_inputs()）
- `.json`: 测试用例文件（JSON Lines格式）


### 2. 算子生成Pipeline（8阶段）

**重要：必须先完成算子生成，再执行benchmark评估！**

对于每个算子，按顺序执行以下8阶段Pipeline生成AscendC算子：

1. **Operator Description Generation** - 生成算子描述JSON
   - 调用 `op-desc-generation` skill
   - 输出: `{op_name}/{op_name}_op_desc.json`

2. **Reference Generation** - 复制参考PyTorch实现
   - 从数据集复制参考实现到output目录
   - 输出: `{op_name}/{op_name}_reference.py`

3. **Functional Conversion** - 转换为Functional API风格
   - 调用 `functional-conversion` skill
   - 输出: `{op_name}/{op_name}_functional.py`

4. **Ascend Project Generation** - 生成Ascend C项目框架
   - 调用 `ascend-call-generation` skill
   - 生成 project.json, cpp绑定代码, python调用代码
   - 创建AscendC项目目录结构

5. **DSL Baseline Generation** - 生成DSL基线代码
   - 调用 `dsl-baseline-generation` skill
   - 输出: `{op_name}/{op_name}_dsl.py`

6. **DSL Lowering** - lowering到Ascend C代码
   - 调用 `dsl-lowering` skill（通过task调用sub-agent）
   - 生成完整的AscendC算子代码

7. **Build \u0026 Install** - 编译并安装算子包
   - 编译生成 `custom_opp_ubuntu_aarch64.run`
   - 安装到 `output/{op_name}/`


**⏱️ 重要：算子生成耗时说明**

算子生成是整个过程**最关键也最耗时**的步骤：
- **生成时间**：单个算子生成通常需要 **20-30 分钟**
- **超时设置**：默认超时时间为 **600秒（10分钟）**，DSL lowering 超时时间为 **1200秒（20分钟）**，可通过 `--timeout` 参数调整，如果超时自动重试
- **耐心等待**：算子生成是后续所有步骤的基础，必须确保完成，不可跳过


### 3. 执行Benchmark评估（算子生成完成后）

**只有在所有算子的8阶段Pipeline完成后，才执行benchmark评估！**




1. 验证PyBind安装 Check custom_ops installation

```bash
$ pip list | grep custom-ops
custom-ops                        1.0
```
如果没有看到 `custom-ops`，请检查安装步骤是否正确完成，确保 `.run` 包已正确安装。调用`.opencode/skills/ascend_benchmark_evaluator/scripts/generate_pybind.py` 生成PyBind代码，并重新安装算子包。


2. **Environment Setup** - 设置环境变量
   - 设置 `ASCEND_CUSTOM_OPP_PATH`
   - 更新 `LD_LIBRARY_PATH`

3. 执行评估脚本
```python3
python3 .opencode/skills/ascend_benchmark_evaluator/scripts/eval_operator_generic.py --op {op_name} 2>&1 | tail -30
```
4. 输出测评结果到 `${pwd}/output/benchmark_report_{timestamp}/` 目录下，包含 `benchmark_report.json` 和 `benchmark_summary.md`
每生成一个算子评估结果，都会增量更新报告目录下的 `benchmark_report.json` 和 `benchmark_summary.md`，以便持续跟踪评估进度和结果。

### 4. 输出结果

**重要：所有输出必须在 `${pwd}/output` 目录下（Agent约束）**

**关键设计**：算子目录必须直接在 `${pwd}/output/{op_name}/` 下（与其他 skills 兼容）

评估完成后生成：
```
${pwd}/output/
├── GELU/                           # 算子目录（直接放在output下）
│   ├── GELU_reference.py
│   ├── GELU_custom.py
│   └── vendors/customize/          # 算子环境依赖
├── SwiGLU/                         # 算子目录
│   └── ...
└── benchmark_report_{timestamp}/   # 报告目录
    ├── benchmark_report.json       # 详细评估结果
    └── benchmark_summary.md        # 可读性报告
```

**目录约束**：
- **算子目录**：必须在 `${pwd}/output/{op_name}/` 下（以便其他 skills 能找到）
- **报告目录**：`${pwd}/output/benchmark_report_{timestamp}/`
- 如果指定 `--output` 参数不在 `${pwd}/output` 下，会自动调整为 `${pwd}/output/{name}`

### 3. 循环阶段
   - 如果有新的算子需要评估，回到阶段1执行生成，再执行阶段2评估
   - 这样可以持续跟踪新算子的评估结果，保持报告的更新

## API Usage

### 阶段1：算子生成（必须先执行）

使用各个skill按顺序生成算子：

```python
# 示例：为单个算子执行完整7阶段Pipeline
# 注意：实际使用时应该通过sub-agent或task并行处理多个算子

# Stage 1: Operator Description Generation
load_skill("op-desc-generation")
# 生成 {op_name}_op_desc.json

# Stage 2: Reference Generation  
# 复制参考实现到output目录

# Stage 3: Functional Conversion
load_skill("functional-conversion")
# 生成 {op_name}_functional.py

# Stage 4: Ascend Project Generation
load_skill("ascend-call-generation")
# 生成 project.json, cpp代码, python代码

# Stage 5: DSL Baseline Generation
load_skill("dsl-baseline-generation")
# 生成 {op_name}_dsl.py

# Stage 6: DSL Lowering（重要：使用task调用sub-agent）
launch_task("dsl-lowering", op_name)
# 生成完整AscendC代码

# Stage 7: Build 
# 编译并安装算子包
```

### 阶段2：Benchmark评估（算子生成完成后）

**重要：只有在7阶段Pipeline完成后，才执行以下命令！**

1.**验证PyBind安装** Check custom_ops installation

```bash
$ pip list | grep custom-ops
custom-ops   

source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
export ASCEND_CUSTOM_OPP_PATH=/home/lvyou/y00889327/OpenOps-main/output/softmax/vendors/customize && \
source /home/lvyou/y00889327/OpenOps-main/output/softmax/vendors/customize/bin/set_env.bash && \
python3 -c "import torch; import torch_npu; import custom_ops_lib; print('Available ops:', [a for a in dir(custom_ops_lib) if not a.startswith('_')])"
Available ops: ['softmax_custom'] 
```
如果没有看到 `custom-ops`，请检查安装步骤是否正确完成，确保 `.run` 包已正确安装。调用`.opencode/skills/ascend_benchmark_evaluator/scripts/generate_pybind.py` 生成PyBind代码，并重新安装算子包。

```python3
python3 .opencode/skills/ascend_benchmark_evaluator/scripts/generate_pybind.py  {op_name}
```
如果查看custom_ops_lib时输出的算子softmax_custom不是目标算子，可能需`pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl`安装生成的pybind包，确保算子可用。


2.**Environment Setup** - 设置环境变量
   - 设置 `ASCEND_CUSTOM_OPP_PATH`
   - 更新 `LD_LIBRARY_PATH`

示例:
```bash
$ cd /home/lvyou/y00889327/OpenOps-main && \
export ASCEND_CUSTOM_OPP_PATH=${pwd}/output/Add/vendors/customize && \
source ${pwd}/output/Add/vendors/customize/bin/set_env.bash && \
export LD_LIBRARY_PATH=${pwd}/output/Add/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
```


3. 执行评估脚本
```
python3 .opencode/skills/ascend_benchmark_evaluator/scripts/eval_operator_generic.py --op {op_name} 2>&1 | tail -30
```
4. 输出测评结果到 `${pwd}/output/benchmark_report_{timestamp}/` 目录下，包含 `benchmark_report.json` 和 `benchmark_summary.md`
每生成一个算子评估结果，都会增量更新报告目录下的 `benchmark_report.json` 和 `benchmark_summary.md`，以便持续跟踪评估进度和结果。


## 输出字段说明

### benchmark_report.json

```json
{
  "summary": {
    "total_operators": 50,
    "successful": 45,
    "failed": 5,
    "success_rate": 0.9,
    "avg_performance_speedup": 1.15
  },
  "operators": [
    {
      "name": "softmax",
      "status": "success",
      "correctness": {
        "passed": true,
        "match_rate": 100.0,
        "max_diff": 1.86e-09
      },
      "performance": {
        "ref_time_ms": 0.090,
        "custom_time_ms": 0.084,
        "speedup": 1.07
      },
      "pipeline_stages": {
        "op_desc": "success",
        "reference": "success",
        "functional": "success",
        "ascend_project": "success",
        "dsl_baseline": "success",
        "dsl_lowering": "success",
        "build": "success",
        "evaluate": "success"
      },
      "error": null
    }
  ]
}
```
### 阶段3：循环阶段
   - 如果有新的算子需要评估，回到阶段1执行生成，再执行阶段2评估
   - 这样可以持续跟踪新算子的评估结果，保持报告的更新

## 配置说明


**NPUKernelBench标准格式不需要config.json**，算子会自动从文件名解析（如 `1_GELU.py` → `GELU`）。

## 错误处理

| 错误类型 | 处理方式 |
|---------|---------|
| Pipeline阶段失败 | 记录失败阶段，继续下一个算子 |
| 编译错误 | 标记为失败，记录错误日志 |
| 正确性不匹配 | 记录匹配率，标记为失败 |
| 超时 | 终止当前算子，标记为超时 |

## Note

### 重要：三阶段执行流程

**必须严格按照以下三阶段顺序执行：**

1. **阶段1：算子生成** 
   - 对每个算子执行完整的8阶段Pipeline
   - 使用各个skills (`op-desc-generation`, `functional-conversion`, `ascend-call-generation`, `dsl-baseline-generation`, `dsl-lowering`)
   - `dsl-lowering` 必须使用task调用sub-agent执行
   - 确保每个算子目录下有完整的AscendC实现（包括 `.run` 安装包）

2. **阶段2：Benchmark评估**
   - **只有**在阶段1完成后，才执行 `eval_operator_generic.py`
   - 评估正确性和性能，生成报告，增量输出到 `${pwd}/output/benchmark_report_{timestamp}/`

3. **阶段3：循环阶段1和阶段二**
   - 如果有新的算子需要评估，回到阶段1执行生成，再执行阶段2评估
   - 这样可以持续跟踪新算子的评估结果，保持报告的更新


### 其他注意事项

- Benchmark执行时间较长，建议使用screen/tmux保持会话
- 每个算子独立输出到 `output/{op_name}/` 目录
- 支持断点续跑：已完成的算子会跳过
- `dsl-lowering` skill **必须**通过task调用sub-agent执行，不能直接在主agent中调用

Base directory for this skill: file:///home/lvyou/y00889327/OpenOps-main/.opencode/skills/ascend_benchmark_evaluator
Relative paths in this skill (e.g., scripts/, reference/) are relative to this base directory.
