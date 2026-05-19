# AscendC Debug — 归档与退出协议

> **读取时机**：Gate-V 返回信号后立即 Read 本文件：
> - **CONTINUE** → 执行「归档当前轮次」后 `attempt += 1`，回到 Step 1
> - **PASS** → 执行 Step 5 成功收尾
> - **STOP**（非 PASS）→ 执行 Step 6 失败报告
> - **所有结局**退出前必须执行 Step 7，产出 `debug_trace.md` + `debug_status.json`

---

### 归档当前轮次 (CONTINUE 时执行)

**每次归档时，比较当前轮 match_rate 与历史最佳，决定是否更新最佳代码。**

```bash
# 1. 保存本轮取证报告和审计报告
mkdir -p "{task_dir}/precision_tuning/history/attempt_{attempt}"
cp "{task_dir}/precision_tuning/forensics_report_{attempt}.json" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/forensics_report.json"
cp "{task_dir}/precision_tuning/precision_audit_{attempt}.md" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/precision_audit.md"

# 2. 更新最佳代码
current_mr=$(python3 -c "import json; r=json.load(open('{task_dir}/precision_tuning/validation_result_attempt_{attempt}.json')); print(r.get('match_rate', '0'))")
best_mr=0
if [ -f "{task_dir}/precision_tuning/history/current_best/match_rate.txt" ]; then
    best_mr=$(cat "{task_dir}/precision_tuning/history/current_best/match_rate.txt")
fi

is_better=$(python3 -c "print('yes' if float('$current_mr') >= float('$best_mr') else 'no')")
if [ "$is_better" = "yes" ]; then
    mkdir -p "{task_dir}/precision_tuning/history/current_best/code_snapshot"
    cp -r "{task_dir}/kernel/" \
       "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/"
    cp "{task_dir}/model_new_ascendc.py" \
       "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py"
    echo "$current_mr" > "{task_dir}/precision_tuning/history/current_best/match_rate.txt"
    echo "精度改善: $best_mr → $current_mr，已更新最佳代码"
fi

# 3. 保存下一轮的起始快照
mkdir -p "{task_dir}/precision_tuning/history/attempt_{next_attempt}/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/model_new_ascendc.py"
```

然后 `attempt += 1`, 回到 Step 1。

---

### Step 7: 退出前强制产物（所有分支 / 所有结局共用）

> **本 Step 是 Step 5 / Step 6 的共同前置**：无论 session 以 `success` / `failed` / `stopped_by_gate` / `stopped_by_loop_limit` / `progressed_to_new_failure_type` / `timeout` / `skipped_env_issue` / `skipped_unsupported_type` 中哪种结局退出，**必须**在退出前产出以下两份文件。任一缺失将导致本次 debug 叙事丢失、下游无法判定结果。

#### 7.1 `{task_dir}/debug_trace.md`（详细叙事，4 节强制 + 可选附录）

**原则**：详细程度对标 `trace.md`，但只强制**有可靠数据源的 section**（findings §3.2.6）。

```markdown
# AscendC Debug Trace

## 1. 调用入口快照（强制）
- 调用时间: <ISO timestamp>
- task_dir: {task_dir}
- session_branch: <1-P / 1-B / 1-I / 1-R / 1-T>
- 入口 `.verify_status/latest.json` 完整快照（由主 agent 产出或本 agent Initialization Protocol Step B 生成）
- 主 agent `{task_dir}/trace.md` 摘要（可选：引用最后一次 AscendC 失败的错误 tail）
- 进入时 kernel/ 基线快照路径（`precision_tuning/history/baseline/code_snapshot`）

## 2. 迭代历史（强制，每轮一节）

### Attempt 0
- 进入时 verify_status 关键字段: failure_type, failed_step, duration_sec, exit_code
- 诊断摘要: 引用 audit_0.md（或对应分支 audit 文件）的摘要 section
- 修复代码改动: 修改文件列表 + 函数 / 行号级 diff 摘要（不贴全文）
- Gate-通用: PASS / FAIL + 未通过项
- Gate-分支 (F/A/V): PASS / FAIL + 关键数值
  - 1-P: mismatch_ratio / max_abs_diff 变化
  - 1-B: failed_step 推进情况
  - 1-I: import.status 变化
  - 1-R: crash signal / crash 位置变化
  - 1-T: duration_sec 变化
- 本轮退出 verify_status 快照
- outcome: passed / improved / stagnant / regressed

### Attempt 1 ... N（同上）

## 3. 最终 Verdict（强制）
- session_outcome: success / failed / stopped_by_gate / stopped_by_loop_limit / progressed_to_new_failure_type / timeout / skipped_env_issue / skipped_unsupported_type / crashed
- 退出时 verify_status 快照
- 若 success: 确认全量 `.json.bak` 恢复后 verify 通过
- 若 failed / stopped_*: 明确原因
- 若 progressed_to_new_failure_type: 新 failure_type 是什么（需用户独立再触发本 agent 才能进入新分支）

## 4. 产物清单（强制）
- 各轮 audit 文件相对 {task_dir} 路径
- tuning_directions.json（精度分支）或对应分支的方向记录文件
- history/baseline/code_snapshot/ 和各轮 attempt_N/code_snapshot/
- .verify_status/phase8_attempt_*.json
- debug_status.json 路径

## 附录 A: 走偏点（可选但推荐）
- 尝试失败的修复方向摘要（对应 tuning_directions.json outcome ∈ {stagnant, regressed}）
- 平台 / API 限制 workaround
- 反作弊触发记录

## 附录 B: 知识库检索记录（仅精度分支 + 其他分支若有）
- search 调用次数 + 主要关键词
- 命中的 knowledge entries

## 附录 C: 耗时细分（可选）
- 总 wall clock（classify_verify_result 各轮 JSON 的 started_at/ended_at 差值之和 + session 整体运行时间）
- 若能区分 Step 级耗时则列出，不能则写"未精细记录"
```

**强制要求**（findings §3.2.6）：
- 只强制 4 节（入口快照 / 迭代历史 / Verdict / 产物清单），其余为附录
- 第 2 节每一轮都不可省略（含 attempt 0 到最后一轮）
- 第 4 节产物清单必须是 `{task_dir}` 下的**相对路径**
- 中文为主，代码 / 路径 / 识别符用英文；Markdown 层级严格 `## 1. ... ## 2. ...`；JSON 快照用 fenced code block 内嵌

#### 7.2 `{task_dir}/debug_status.json`（机器可读 verdict）

```json
{
  "schema_version": 1,
  "session_outcome": "success | failed | stopped_by_gate | stopped_by_loop_limit | progressed_to_new_failure_type | timeout | skipped_env_issue | skipped_unsupported_type | crashed",
  "session_branch": "1-P | 1-B | 1-I | 1-R | 1-T",
  "started_at": "<ISO>",
  "ended_at": "<ISO>",
  "attempts_used": "<int>",
  "entry_failure_type": "<进入时的 final_status.failure_type>",
  "final_failure_type": "<从最后一次 .verify_status/phase8_attempt_{N}.json 读取>",
  "final_verify_status_path": "{task_dir}/.verify_status/phase8_attempt_{N}.json",
  "notes": "<一句话说明，例：首轮 COMPILE_ERROR 已定位为 template_arg_fix，attempt 1 Gate-V 推进至 import>"
}
```

**字段约束**：
- `schema_version = 1`（本版本固定）
- `session_outcome` 必须是上列 9 种之一；其他值视为未知结局，下游消费者应视同 `crashed`
- `session_branch` 必须与 Step 0.3 锁定的分支一致
- `started_at` / `ended_at` 用 ISO 8601（带 UTC 时区）
- `final_failure_type` 从**最后一次**本 session 内触发的 `utils/verification_ascendc.py` + `utils/classify_verify_result.py` 产出读取；若一次都没跑（例如 `skipped_*`），与 `entry_failure_type` 一致
- `final_verify_status_path` 也指向最后一次本 session 内的 phase8_attempt_N.json；未跑时为 `null`

#### 7.3 硬约束重申

- ⛔ **禁止 append 或重写 `{task_dir}/trace.md`** —— 主 agent 产物，本 skill 全程只读；所有 debug 叙事 / verdict 都落到 `debug_trace.md` + `debug_status.json`
- ⛔ **禁止修改 `utils/` / `CMakeLists.txt` / `setup.py` / `agents/` / `skills/`** —— 只能改 `{task_dir}/kernel/` 下文件，`{task_dir}/precision_tuning/` 下写 skill 产物
- ⛔ **禁止删除或重写 `{task_dir}/.verify_status/latest.json`、`{task_dir}/{op_name}.json.bak`** —— 上游 artefact / 全量用例备份，只读
- 写完 `debug_trace.md` + `debug_status.json` 后，再进入 Step 5（成功）或 Step 6（失败）的**报告输出**部分；Step 5 / 6 里的"归档当前轮 / 更新 current_best / 全量验证"等动作在写 Step 7 产物前完成即可（Step 7 是退出前的最后一步，只负责 debug_trace / debug_status 两份产物）

---

### Step 5: 成功收尾

精度通过后:

**5.0 全量用例验证（若存在 `.json.bak`，则为强制步骤）**

若 `{task_dir}/{op_name}.json.bak` 存在，说明当前 `{op_name}.json` 为精简用例；此时必须先恢复全量用例，再做一次最终验证：

```bash
if [ -f "{task_dir}/{op_name}.json.bak" ]; then
    cp "{task_dir}/{op_name}.json.bak" "{task_dir}/{op_name}.json"
    bash skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh {task_name}
fi
```

处理规则：
- 若 `.json.bak` 不存在：跳过本步骤，直接进入 5.1
- 若全量验证通过：继续进入 5.1
- 若全量验证失败：**不得**宣布成功；仅允许继续修改 `{task_dir}/kernel/` 下文件，并重新执行全量验证
- 全量验证失败后的补救次数最多 3 次（含首次全量验证）；超过次数仍失败，则转 Step 6 失败报告
- 若做过全量验证，最终成功报告中的 `final_match_rate` / `final_max_diff` 应以全量验证结果为准，而不是精简验证结果

建议额外保存全量验证结果：

```json
{task_dir}/precision_tuning/full_validation_result_attempt_{attempt}.json
{
  "attempt": "<N>",
  "used_full_cases": true,
  "correctness_passed": "true/false",
  "evaluate_stdout": "<全量 evaluate_ascendc.sh 完整输出>",
  "match_rate": "<从 stdout 提取>",
  "max_diff": "<从 stdout 提取>"
}
```

**5.1 归档当前轮次 + 更新 current_best（最终 PASS 时必须执行）:**
```bash
# 归档本轮取证报告和审计报告
mkdir -p "{task_dir}/precision_tuning/history/attempt_{attempt}"
cp "{task_dir}/precision_tuning/forensics_report_{attempt}.json" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/forensics_report.json"
cp "{task_dir}/precision_tuning/precision_audit_{attempt}.md" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/precision_audit.md"

# 更新 current_best 为最终通过的代码
mkdir -p "{task_dir}/precision_tuning/history/current_best/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py"
echo "100.0" > "{task_dir}/precision_tuning/history/current_best/match_rate.txt"
echo "精度通过，current_best 已更新为 100.0"
```

**5.2 生成候选知识库条目 (Agent 执行):**

基于 [ROOT_CAUSE] 和 [FIX_PLAN]，生成一条知识库候选条目，写入：
`{task_dir}/precision_tuning/candidate_kb_entry.json`

格式要求:
```json
{
  "title": "<标准化中文标题，含英文关键词，如：LayerNorm 尾块 Padding 污染精度>",
  "feature": "<错误特征签名，泛化表达，不要写死具体 shape 或 tile size，如：tail_spike 模式，尾块 mismatch 率显著高于主体>",
  "reason": "<深层原因，50-200字，描述为什么会出现此问题>",
  "fix": "<通用修复指南，50-200字，描述应该如何修复，不要包含具体行号>",
  "type": "<FIX_PRECISION_XXX 枚举值，与 [FIX_PLAN] 中的修复类型一致>"
}
```

注意:
- `title` 必须含英文关键词（供 RAG 检索），格式为"中文描述 (English Keywords)"
- `feature` 要泛化，不要写 `last_dim=37` 或 `tile_size=128` 这种具体值
- `fix` 要通用，不要引用具体代码行号或变量名
- `type` 必须从以下枚举中选择：FIX_PRECISION_PADDING / FIX_PRECISION_TAIL / FIX_PRECISION_REDUCTION / FIX_PRECISION_TYPECAST / FIX_PRECISION_LAYOUT / FIX_PRECISION_SYNC / FIX_PRECISION_OVERFLOW / FIX_PRECISION_LOGIC / FIX_PRECISION_OTHER

**5.2.5 相似性检查 (Python 执行):**
```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_knowledge.py check \
    --kb-path skills/ascendc/ascendc-debug/references/precision_knowledge_base.json \
    --candidate-path {task_dir}/precision_tuning/candidate_kb_entry.json
```

输出 JSON 到 stdout，摘要到 stderr，关键字段：
- `similar_entries`: 相似度 >= 0.10 的已有条目列表（含 title/feature/reason/fix，按 score 降序）
- `suggestion`: `"new"`（无相似）或 `"review_needed"`（有相似，需 Agent 判断）

**5.2.6 写入决策 (Agent 执行):**

读取 5.2.5 的输出，对 `similar_entries` 做语义判断，确定 `action`：

| suggestion | 语义判断结果 | action | 需要额外操作 |
|---|---|---|---|
| `new` | 无相似条目 | `new` | 无 |
| `review_needed` | 候选与已有条目完全重叠，无新信息 | `abandon` | 无 |
| `review_needed` | 候选有新细节（新触发场景、更精确的 fix、补充的 op_type 等） | `merge` | **必须先更新 `candidate_kb_entry.json`**（见下） |
| `review_needed` | 关键词重叠但根因/场景本质不同 | `new` | 无 |

> **merge 操作前 Agent 必须先丰富 `candidate_kb_entry.json`**：将相似条目的已有内容与新候选内容合并，形成更完整的条目（保留旧条目的核心知识，补充新触发场景或 fix 细节），再写回 `{task_dir}/precision_tuning/candidate_kb_entry.json`。Python 脚本只负责将该文件内容替换到知识库对应位置，合并本身由 Agent 完成。

**5.3 写入知识库 (Python 执行):**
```bash
# action 由 Step 5.2.6 决策确定（new / merge / abandon）
# action=merge 时追加 --merge-target-title，值为被替换条目的完整 title（精确匹配）
python3 skills/ascendc/ascendc-debug/scripts/precision_knowledge.py dump \
    --kb-path skills/ascendc/ascendc-debug/references/precision_knowledge_base.json \
    --task-name {task_name} \
    --op-name {op_name} \
    --action {action} \
    [--merge-target-title "<existing entry title>"]
```

**5.4 保存成功代码快照:**
```bash
# 将最终通过代码保存到 history/success/（永久保留，不覆盖）
mkdir -p "{task_dir}/precision_tuning/history/success/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/success/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/success/code_snapshot/model_new_ascendc.py"
echo "成功代码已保存到 history/success/code_snapshot/"
```

> **从最佳代码恢复（如需重新调优）：**
> ```bash
> cp -r "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```

**5.5 输出成功报告:**
```
[PRECISION_TUNING_RESULT]
  status: SUCCESS
  attempts: <总轮次>
  final_match_rate: <最终 match rate，若跑过全量则取全量结果>
  final_max_diff: <最终 max diff，若跑过全量则取全量结果>
  root_cause_summary: <一句话总结根因>
  fix_summary: <一句话总结修复内容>
```

---

### Step 6: 失败报告

如果 Gate-V 返回 STOP:

输出失败报告, 包含所有轮次的历史:
```
[PRECISION_TUNING_RESULT]
  status: FAILED
  attempts: <总轮次>
  loop_stop_reason: <Gate 给出的停止原因>
  history:
    attempt 0: hint=<pattern>, mismatch=<ratio>, fix=<一句话>
    attempt 1: hint=<pattern>, mismatch=<ratio>, fix=<一句话>
    ...
  remaining_issue: <当前仍存在的问题描述>
  suggestion: <给人工分析的建议>
```

> **注意:** 失败时 `history/current_best/` 中保存了精度最好的代码。如需以此为基础重新调优，恢复方法：
> ```bash
> cp -r "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```
> 如需恢复到最初基线：
> ```bash
> cp -r "{task_dir}/precision_tuning/history/baseline/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/baseline/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```
