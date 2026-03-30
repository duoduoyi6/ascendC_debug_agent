# AscendOpGenAgent 贡献规范

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

### 1.2 合入门禁

所有 PR 合入前须通过双通路冒烟和 Benchmark 评测两项门禁，确保主干代码质量不退化。

#### 1.2.1 双通路冒烟测试

**所有非文档 PR** 须通过 op-router agent 分别调用 Triton / AscendC 侧 agent，端到端跑通（编译通过 + 精度正确）。

**执行方式**：使用 `op-router` agent 在 OpenCode 中执行评测：

- **Triton 通路**：调用 `AKG-triton` Agent 生成一个基础算子（如 level1 problem1），验证编译通过 + 精度正确
- **AscendC 通路**：调用 `lingxi-ascendc` Agent 生成一个基础算子（如 level1 problem1），验证编译通过 + 精度正确

两条通路都通过后在 PR 模板中标记结果。若因环境原因无法运行，须在 PR 中说明。

> **退化豁免**：PR 中写明原因 + follow-up plan，2 个 maintainer Approve 后可豁免。
> 文档 / 基础设施类 PR 仅需不影响代码功能，无需上述门禁。



#### 1.2.2 Benchmark 评测

跑完 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md) 全部任务，精度性能不劣化。

| 门禁 | 含义 | 算子生成 | 性能优化 | 其他 |
|------|------|:---:|:---:|:---:|
| 通过率不退化 | 编译/精度通过数 >= BASELINE | 必须 | 必须 | - |
| Speedup 不退化 | 平均值及逐任务 >= BASELINE × 0.95 | 必须 | 必须 | - |
| 性能有提升 | 至少 1 个算子加速比提升 >= 5% | - | 必须 | - |

> "其他"指 Benchmark、框架改动、Bug 修复、文档等类型的 PR。

**执行方式**：使用 `benchmark-Scheduler` agent 在 OpenCode 中执行评测：

```text
评测KernelBench中Level 1的2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 57, 61, 63, 64, 67, 82, 87, 99, 100和Level 2的6, 12, 17, 23, 30, 94的任务,agent_workspace是<你的AscendOpGenAgent路径>
```

评测完成后，将 `agent_report.md` 中的数据与 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md) 逐项对比，在 PR 模板中填写结果。


### 1.3 基线管理

维护 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md)，记录 main 分支最新评测结果。

**更新规则**：算子生成/性能优化 PR 合入 main 后，若性能提升须更新 BASELINE.md（日期、数据、通过率、平均 Speedup）。

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

---

## 二、新功能接入流程

1. **创建 Agent**：`agents/<name>.md`
2. **创建 Skill**：`skills/<skill-name>/`
3. **注册到 op-router**：在 [`agents/op-router.md`](agents/op-router.md) 添加路由分支（PR 待合入，暂未上仓）
4. **添加 Benchmark Case**：至少 5 个 case
5. **提供基线数据**：跑一轮 benchmark，追加到 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md)
6. **提交 PR**：按第一节规范填写模板
