# Proposal 001: 补充 sigmoid 高阶 API 文档并引导 agent 优先使用硬件原语

- 日期: 2026-04-06
- 基于 trace: swiglu_20260406
- 影响阶段: 阶段一 (TileLang) / 阶段二 (AscendC) / 指导文档

## 问题诊断

### 失败现象

全量用例验证中 9/50 case 失败（阶段四），失败模式为少量元素（< 1.5%）超出 atol=0.01 容差，max\_abs\_diff 最高达 1212.87。失败集中在较大 tensor 上，涉及 float16 和 float32 两种 dtype。

### Agent 的实际行为路径

Agent 将 SwiGLU 的 silu(a) 分解为手动 sigmoid 链：
```
Muls(tmp, a, -1.0f)     → tmp = -a
Exp(tmp, tmp)            → tmp = exp(-a)
Adds(tmp, tmp, 1.0f)    → tmp = 1 + exp(-a)
Reciprocal(tmp, tmp)     → tmp = sigmoid(a)
Mul(tmp, a, tmp)         → tmp = silu(a)
```

这条链在数学上正确，但在硬件上存在精度累积问题：每一步 Ascend 硬件实现都有微小误差，4 步链式计算误差累积放大，尤其在 exp 值域边界（|a| > 10）处偏差明显。

### 根因分析

1. **TileLangAscendProgrammingGuide.md 缺少 `T.tile.sigmoid` 文档**。Section 6.1 的数学运算表列出了 exp、reciprocal、relu、leaky\_relu、sin、cos 等，但遗漏了 `sigmoid`。Agent 在参考文档中找不到 sigmoid API，只能手动分解。

2. **`T.tile.sigmoid(dst, src, tmp)` 确实存在于 `TileLang-AscendC-API-Mapping.md` 中**（第 38 行），映射到 `AscendC::Sigmoid(...)`。但 AGENT\_TILELANG.md 明确禁止 agent 读取该文件（"禁止读取 `docs/TileLang-AscendC-API-Mapping.md`"），因此 agent 在 TileLang 阶段无法得知 sigmoid API 的存在。

3. **AscendC::Sigmoid 是硬件高阶 API**（见 `docs/AscendC_knowledge/api_reference/pages/atlasascendc_api_07_0793.md`），内部实现使用了数值稳定的近似算法，精度显著优于手动 exp+reciprocal 链。

4. **缺少通用指导**：harness 文档中没有"优先使用硬件提供的高阶数学原语（sigmoid、tanh 等），避免手动分解为基础运算"这一原则。

### 反事实推理

如果 `TileLangAscendProgrammingGuide.md` 中列出了 `T.tile.sigmoid`，agent 大概率会直接使用它（仅需 1 步调用 + 1 步 Mul），而非手写 4 步链。对应的 AscendC 转译也会自然使用 `AscendC::Sigmoid`，从而避免精度问题。

## 修改建议

### 修改 1: `docs/TileLangAscendProgrammingGuide.md`

**位置**: Section 6.1 Math and Logical Ops 表格，在 Leaky ReLU 行之后

**原文**:
```
| Leaky ReLU | `T.tile.leaky_relu(dst, src0, scalar)` | elementwise Leaky ReLU, `dst = src0 if src0 >= 0 else src0 * scalar` |
| AXPY | `T.tile.axpy(dst, src0, scalar)` | fused axpy, `dst = scalar * src0 + dst` |
```

**建议改为**:
```
| Leaky ReLU | `T.tile.leaky_relu(dst, src0, scalar)` | elementwise Leaky ReLU, `dst = src0 if src0 >= 0 else src0 * scalar` |
| Sigmoid | `T.tile.sigmoid(dst, src0, tmp)` | elementwise sigmoid, `dst = 1 / (1 + exp(-src0))`; `tmp` is a UB uint8 buffer for internal workspace |
| AXPY | `T.tile.axpy(dst, src0, scalar)` | fused axpy, `dst = scalar * src0 + dst` |
```

**理由**: 补全遗漏的 sigmoid API 文档，使 agent 在 TileLang 阶段能发现并使用硬件优化的 sigmoid 实现。

### 修改 2: `docs/TileLangAscendProgrammingGuide.md`

**位置**: Section 1.1 Programming Guidelines 末尾

**原文**:
```
- Use `T.tile.broadcast` sparingly because it can consume large UB temporary space, and prefer row-wise or column-wise tile compute patterns when UB is constrained.
```

**建议改为**:
```
- Use `T.tile.broadcast` sparingly because it can consume large UB temporary space, and prefer row-wise or column-wise tile compute patterns when UB is constrained.
- Prefer high-level math primitives (`T.tile.sigmoid`, `T.tile.sin`, `T.tile.cos`) over manual decomposition into basic ops (`exp` + `reciprocal` etc.). Hardware primitives are numerically more stable and often faster than multi-step chains.
```

**理由**: 建立通用原则，引导 agent 在所有算子任务中优先使用高阶原语，而非仅对 SwiGLU 有效。

### 修改 3: `docs/dsl2Ascendc.md`

**位置**: 第五章 常见陷阱速查表末尾新增一行

**原文**:
```
| 使用不存在的 Divs | 编译报错 | 改用 `Muls(dst, src, 1.0f/scaleVal, count)` |
```

**建议改为**:
```
| 使用不存在的 Divs | 编译报错 | 改用 `Muls(dst, src, 1.0f/scaleVal, count)` |
| 手动分解 sigmoid 为 exp+reciprocal | 精度不达标 | 使用 `AscendC::Sigmoid(dst, src, tmpBuf, count)` 高阶 API，需额外 uint8 临时空间 |
```

**理由**: 在 AscendC 阶段的陷阱表中明确标注此常见错误模式，帮助 agent 在转译阶段也避免手动分解。

## 预期效果

- Agent 在实现包含 sigmoid 的算子（SwiGLU、Swish、GELU-sigmoid 近似等）时，直接使用 `T.tile.sigmoid` / `AscendC::Sigmoid`
- 减少 TileLang 和 AscendC 中的计算步骤（4 步 → 1 步），降低精度误差和 UB 使用
- 全量用例验证的失败 case 预期归零或大幅减少

## 风险评估

- **低风险**: 修改仅增加文档内容和指导原则，不删除或更改已有信息
- **泛化性好**: "优先使用高阶原语"原则适用于所有算子任务，不会过拟合 SwiGLU
- **潜在副作用**: 如果某个算子确实需要自定义 sigmoid 变体（如带 scaling 的 sigmoid），agent 可能过度依赖标准 API。但 "prefer" 措辞留有灵活空间

## 备注

- `T.tile.sigmoid` 的 `tmp` 参数大小需通过 `GetSigmoidMaxMinTmpSize` 获取（AscendC 侧），TileLang 侧的 tmp 大小约定需确认是否已有自动推导或需手动指定
- 精简 case 集（阶段零）未能捕获全量验证中的 9 个失败，但这属于随机输入值分布问题而非 case 设计缺陷——相同 shape 在不同随机种子下可能通过或失败。核心问题仍是计算精度，非 case 覆盖度
