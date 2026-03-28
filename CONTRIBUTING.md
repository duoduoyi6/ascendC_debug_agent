# AscendOpGenAgent 贡献规范

本文档定义 AscendOpGenAgent 仓库的合入规范和格式规范。所有贡献者请遵循。

---

## 一、PR 合入规范

### 1.1 PR 标题格式

```
[<scope>] <描述>
```

| scope | 适用场景 |
|-------|---------|
| `triton` | Triton Ascend 侧改动 |
| `ascendc` | AscendC 侧改动 |
| `benchmark` | Benchmark case / 评测逻辑 |
| `router` | op-router / 路由逻辑 |
| `infra` | CI、脚本、构建 |
| `docs` | 文档 |

示例：
- `[triton] 新增 layernorm 算子生成支持`
- `[ascendc] dsl-lowering tiling pass 优化`
- `[benchmark] NPUKernelBench level2 新增 10 case`
- `[router] op-router 增加 CUDA→Ascend 路由分支`

### 1.2 PR 描述模板

创建 PR 时按以下模板填写（同时放在 `.github/pull_request_template.md`）：

```markdown
## 变更说明
- [ ] 新增算子 / Skill
- [ ] 性能优化
- [ ] Bug 修复
- [ ] Benchmark（新增 case / 修改评测逻辑）
- [ ] Agent / 框架改动
- [ ] 文档 / 基础设施

## 影响范围
- [ ] Triton 侧
- [ ] AscendC 侧
- [ ] 共享（router / benchmark-scheduler）

## 性能数据（涉及算子生成/优化时必填）

### Benchmark 回归

| 数据集 | Level | 变更前通过数 | 变更后通过数 | 变更前 avg Speedup | 变更后 avg Speedup |
|--------|-------|-----------|-----------|------------------|------------------|
| KernelBench | level1 | | | | |
| NPUKernelBench | level1 | | | | |

> 变更前数据取 main 分支最新 `benchmarks/BASELINE.md`

### 测试环境
- 设备型号：
- CANN 版本：
- PyTorch 版本：
```

### 1.3 合入门禁

#### 门禁总览

|  | 编译 | 精度 | 性能不退化 | Benchmark 回归 | 格式合规 | 流水线跑通 |
|--|------|------|----------|---------------|---------|----------|
| 算子生成 | 必须 | 必须 | 必须 | 必须 | - | 必须 |
| 性能优化 | 必须 | 必须 | 必须 + 有提升 | 必须 | - | 必须 |
| Benchmark | - | - | - | 不破坏已有 | - | 必须 |
| 框架改动 | - | - | 推荐 | 推荐 | 必须 | 必须 |
| Bug 修复 | 视情况 | 视情况 | 视情况 | 视情况 | - | 推荐 |
| 文档/infra | - | - | - | - | - | - |

#### 算子生成 / 性能优化类 PR（最严格）

| # | 门禁 | 要求 | 判定标准 |
|---|------|------|---------|
| 1 | 编译通过 | 必须 | 生成代码在目标设备编译 0 error |
| 2 | 精度通过 | 必须 | allclose Pass |
| 3 | Benchmark 通过率不退化 | 必须 | 通过数 >= main 分支 BASELINE |
| 4 | Benchmark 平均 Speedup 不退化 | 必须 | avg Speedup >= BASELINE x 0.95（允许 5% 波动） |
| 5 | 端到端流水线 | 必须 | op-router → agent → skill → verify 完整链路无报错 |

**性能优化类额外要求**：
- 至少 1 个算子 Speedup 提升 >= 5%
- 非目标算子波动在 ±5% 以内

**退化豁免**：PR 描述中写明原因 + follow-up plan，2 个 maintainer Approve 后可豁免。

#### 其他类型 PR

| PR 类型 | 门禁要求 |
|---------|---------|
| Benchmark（新增 case） | case 格式合规 + benchmark-scheduler 可调度 + 至少 1 case 完整跑通 |
| Agent / 框架改动 | 格式合规（见第二节） + 端到端 1 case 跑通 + 不破坏另一侧 |
| Bug 修复 | bug 不复现 + 不引入新问题 |
| 文档 / 基础设施 | 不影响代码功能 |

### 1.4 Review 规则

| PR 类型 | 最少 Approve | 特殊要求 |
|---------|------------|---------|
| 算子生成 / 性能优化 | 2 | 至少 1 个对应 DSL 侧 maintainer |
| 框架改动（跨侧） | 2 | 两侧各 1 个 maintainer |
| 其他 | 1 | - |

**Review 重点**：
- 算子生成：生成逻辑正确性、prompt 质量、reference 准确性、性能数据真实性
- 性能优化：优化手段合理性、性能数据可复现性
- Benchmark：case 代表性、baseline 合理性、评测公平性
- 框架改动：格式一致性、路由正确性、向后兼容

**性能数据真实性**：Reviewer 有权要求在指定设备上重跑。提交者必须记录测试环境（设备型号、CANN 版本、PyTorch 版本）。

### 1.5 分支策略

```
main                         # 保护分支，门禁全通过才能合入
+-- feat/<name>              # 功能开发
+-- fix/<issue-number>       # Bug 修复
+-- bench/<name>             # Benchmark 新增
```

- 统一 **Squash and Merge**
- commit message 保持 PR 标题格式
- 合入后自动删除源分支

### 1.6 Benchmark 基线管理

维护 `benchmarks/BASELINE.md`，记录 main 分支最新评测结果：

```markdown
# Baseline

> main @ <commit-hash>, <date>
> Device: Ascend 910B2, CANN x.x.x, PyTorch x.x.x

## KernelBench level1

| # | Problem | 编译 | 精度 | PyTorch (ms) | Generated (ms) | Speedup |
|---|---------|------|------|-------------|----------------|---------|
| 1 | softmax | Pass | Pass | 0.42 | 0.38 | 1.11x |

通过率：xx/100
平均 Speedup：x.xx

## NPUKernelBench level1

（同上格式）
```

**更新规则**：
- 算子生成 / 性能优化 PR 合入 main 后更新 BASELINE.md
- PR 中"变更前"数据以 BASELINE.md 为准
- 纳入版本管理，可追溯历史

---

## 二、统一格式规范

后续新增功能（CUDA→Ascend、CUDA→Triton 等）都按此格式接入，确保一致性和可扩展性。

### 2.1 Agent 格式

文件位置：`agents/<name>.md`

#### Frontmatter（必填）

```yaml
---
name: <kebab-case, 与文件名一致>
version: <semver>
description: <一行说明>
mode: primary | subagent
temperature: 0.1
tools:
  read: true
  edit: true
  bash: true
  question: true    # primary agent 必须
  task: true        # 需要调用子 agent 时
skills:
  - <skill-name>    # 按调用顺序列出
subagents:          # 可选
  - <agent-name>
---
```

#### 正文结构

```markdown
# <Agent 名称>

## 职责
<!-- 一段话说明这个 agent 做什么 -->

## 流水线
<!-- 分阶段描述 -->

### Stage 1: <阶段名>
- **Skill**: `<skill-name>`
- **输入**: ...
- **输出**: ...
- **检查点**: ...（精度校验、编译检查等）

### Stage 2: ...

## 错误处理
<!-- 重试策略、最大重试次数、失败后行为 -->

## 输出目录
<!-- 完整目录结构树 -->
```

#### 必填字段

| 字段 | 要求 |
|------|------|
| `name` | kebab-case，与文件名一致（如 `AKG-triton` 对应 `AKG-triton.md`） |
| `version` | semver 格式 |
| `description` | 一行中文说明 |
| `mode` | `primary` 或 `subagent` |
| `temperature` | 固定 `0.1` |
| `tools` | 列出所需工具 |
| `skills` | 按调用顺序列出 |

### 2.2 Skill 格式

文件位置：`skills/<scope>/<skill-name>/SKILL.md`

#### 目录结构

```
skills/<scope>/<skill-name>/
├── SKILL.md           # 必须
├── references/        # 参考文档（可选）
└── scripts/           # 辅助脚本（可选）
```

#### SKILL.md 模板

```yaml
---
name: <kebab-case, 与目录名一致>
version: <semver>
description: <一行说明>
---
```

```markdown
# <Skill 名称>

## 功能
<!-- 一段话说明 -->

## 输入

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| | | | | |

## 输出

| 文件 | 路径 | 说明 |
|------|------|------|
| | `output/{op_name}/...` | |

## 执行步骤

1. **步骤名**：做什么、用什么工具/脚本
2. ...

## 质量检查

<!-- 本 skill 完成后的验证条件 -->
- 文件存在性检查
- 编译通过
- 精度校验
- ...

## 引用文档

<!-- references/ 下的文件清单及用途 -->
| 文件 | 用途 |
|------|------|
```

#### 必填字段

| 字段 | 要求 |
|------|------|
| `name` | kebab-case，与目录名一致 |
| `version` | semver 格式 |
| `description` | 一行说明 |

### 2.3 Benchmark Case 格式

#### KernelBench（Python module）

文件位置：`benchmarks/KernelBench/level<N>/<problem_id>_<OperatorName>.py`

```python
"""
Problem <id>: <算子名称>
Level: <1-4>
Category: <activation|reduction|normalization|matmul|conv|pooling|loss|math|composite>
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """<算子描述>"""

    def __init__(self, <init_params>):
        super().__init__()
        # ...

    def forward(self, <inputs>) -> torch.Tensor:
        # ...


# 测试维度
batch_size = ...
dim = ...


def get_inputs():
    """返回 forward() 的输入张量列表"""
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    """返回 __init__() 的参数列表"""
    return []
```

**必须包含**：
- `Model` class（含 `__init__` 和 `forward`）
- `get_inputs()` 函数
- `get_init_inputs()` 函数

**文件命名**：`<problem_id>_<OperatorName>.py`（如 `1_GELU.py`、`15_Matmul_for_lower_triangular_matrices.py`）

#### NPUKernelBench（JSON）

文件位置：`benchmarks/NPUKernelBench/level<N>/<problem_id>_<OperatorName>.json`

```json
{
  "operator": "<算子名称>",
  "category": "<分类>",
  "description": "<数学公式描述>",
  "inputs":{
    "input_shape1": [
      {
        "name": "<参数名>",
        "type": "tensor",
        "required": true,
        "dtype": ["float16", "float32"],
        "shape": [[128], [256, 512], [1024, 1024]]
      }
    ],
    "input_shape2": [
      {
        "name": "<参数名>",
        "type": "tensor",
        "required": true,
        "dtype": ["float16", "float32"],
        "shape": [[128], [256, 512], [1024, 1024]]
      }
    ]
  } ,
  "attributes": {
    "<attr_name>": { "type": "bool", "default": false }
  },
  "constraints": "<约束说明>"
}
```

**必填字段**：`operator`、`inputs`（至少 1 个）
**文件命名**：`<problem_id>_<OperatorName>.json`（如 `1_GELU.json`）

---

## 三、新功能接入流程

后续新增功能（如 CUDA→Ascend、CUDA→Triton 转换）按以下步骤接入：

### 3.1 步骤

1. **创建 Agent**：`agents/<name>.md`，按 2.1 格式编写
2. **创建 Skill 目录**：`skills/<scope>/`，每个 skill 按 2.2 格式，scope 与功能对应
3. **注册到 op-router**：在 `agents/op-router.md` 的路由逻辑中添加新分支
4. **添加 Benchmark Case**：至少 5 个 case 到 KernelBench 或 NPUKernelBench
5. **提供基线数据**：跑一轮 benchmark，数据追加到 `benchmarks/BASELINE.md`
6. **提交 PR**：按第一节合入规范填写模板，附性能数据

### 3.2 scope 扩展约定

| scope | 说明 | skill 目录 |
|-------|------|-----------|
| `triton` | Triton Ascend 算子生成 | `skills/triton/` |
| `ascendc` | AscendC 算子生成 | `skills/ascendc/` |
| `cuda2ascend` | CUDA→Ascend 转换（待建） | `skills/cuda2ascend/` |
| `cuda2triton` | CUDA→Triton 转换（待建） | `skills/cuda2triton/` |

每个 scope 独立目录，skill 互不干扰。Agent 在 frontmatter 的 `skills:` 字段声明自己使用的 skill。

### 3.3 命名约定

- **Agent / Skill**：统一 kebab-case（如 `dsl-lowering`、`kernel-verifier`）
- **name 字段**：必须与文件名/目录名一致
- **Benchmark case**：`<problem_id>_<OperatorName>`，OperatorName 用 PascalCase