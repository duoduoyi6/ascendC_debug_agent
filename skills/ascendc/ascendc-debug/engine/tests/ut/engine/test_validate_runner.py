"""test_validate_runner.py — forensics staleness 缓存 (建议A, Tier 1)。

不真跑 precision_forensics.py 子进程: monkeypatch subprocess.run 计数调用次数，验证:
  1. 首轮无缓存 → 跑子进程 (miss)。
  2. 源码 hash 不变 → 第二轮复用上轮 report，不跑子进程 (hit)，且复制成当前 attempt 名。
  3. kernel 源码改动 → hash 变 → 重跑 (miss)。
  4. 新增/删除源文件 → hash 变 → 重跑 (miss)。
  5. ASCENDC_DEBUG_FORENSICS_NO_CACHE=1 → 永不复用。
  6. 上轮 report 已被删 → 即便 hash 命中也回退重跑 (向正确性倾斜)。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine import validate_runner


def _make_task(tmp: Path) -> Path:
    """造一个含 kernel/ + model_new_ascendc.py 的最小任务目录。"""
    (tmp / "kernel").mkdir(parents=True, exist_ok=True)
    (tmp / "kernel" / "op.cpp").write_text("// v1\n", encoding="utf-8")
    (tmp / "kernel" / "op.h").write_text("#pragma once\n", encoding="utf-8")
    (tmp / "model_new_ascendc.py").write_text("x = 1\n", encoding="utf-8")
    (tmp / "precision_tuning").mkdir(parents=True, exist_ok=True)
    return tmp


class _FakeForensics:
    """fake subprocess.run: 计调用次数，并把 report 写到当前 attempt 名 (模拟脚本产出)。"""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir
        self.calls = 0

    def __call__(self, cmd, **kwargs):
        # prebuild step (build_ascendc.py) 不带 --attempt，直接返回成功，不计入 calls。
        if "--attempt" not in cmd:
            class _R:
                returncode = 0
                stderr = ""
            return _R()
        self.calls += 1
        # 从 cmd 里取 --attempt 值，产出对应 report (模拟真实脚本行为)。
        attempt = cmd[cmd.index("--attempt") + 1]
        report = self.task_dir / "precision_tuning" / f"forensics_report_{attempt}.json"
        report.write_text(json.dumps({"attempt": int(attempt)}), encoding="utf-8")

        class _R:
            returncode = 0
            stderr = ""
        return _R()


class TestForensicsCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = _make_task(Path(self._tmp.name))
        self._env_backup = os.environ.pop("ASCENDC_DEBUG_FORENSICS_NO_CACHE", None)

    def tearDown(self) -> None:
        if self._env_backup is not None:
            os.environ["ASCENDC_DEBUG_FORENSICS_NO_CACHE"] = self._env_backup
        else:
            os.environ.pop("ASCENDC_DEBUG_FORENSICS_NO_CACHE", None)
        self._tmp.cleanup()

    def _run(self, attempt: int, fake: _FakeForensics) -> dict:
        with mock.patch.object(validate_runner.subprocess, "run", fake):
            return validate_runner.run_forensics(self.task_dir, attempt=attempt)

    def test_miss_then_hit(self) -> None:
        fake = _FakeForensics(self.task_dir)
        r0 = self._run(0, fake)
        self.assertTrue(r0["success"])
        self.assertEqual(fake.calls, 1)
        self.assertFalse(r0.get("cached"))  # 重跑路径显式 cached=False

        # 源码未变 → 第二轮复用，不跑子进程。
        r1 = self._run(1, fake)
        self.assertTrue(r1["success"])
        self.assertEqual(fake.calls, 1, "源码不变应复用，不应再跑子进程")
        self.assertTrue(r1.get("cached"))
        self.assertEqual(r1.get("cached_from_attempt"), 0)
        # 复用应把上轮 report 复制成当前 attempt 名 (Gate-F 精确文件名读取)。
        self.assertTrue((self.task_dir / "precision_tuning" / "forensics_report_1.json").exists())

    def test_src_change_triggers_rerun(self) -> None:
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self.assertEqual(fake.calls, 1)

        # 改 kernel 源码 → hash 变 → 重跑。
        (self.task_dir / "kernel" / "op.cpp").write_text("// v2 changed\n", encoding="utf-8")
        r1 = self._run(1, fake)
        self.assertEqual(fake.calls, 2, "源码改动应重跑")
        self.assertFalse(r1.get("cached"))

    def test_add_file_triggers_rerun(self) -> None:
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self.assertEqual(fake.calls, 1)

        # 新增源文件 → hash 变 (文件名并入 hash) → 重跑。
        (self.task_dir / "kernel" / "extra.cpp").write_text("// new\n", encoding="utf-8")
        self._run(1, fake)
        self.assertEqual(fake.calls, 2, "新增源文件应重跑")

    def test_no_cache_env_disables(self) -> None:
        os.environ["ASCENDC_DEBUG_FORENSICS_NO_CACHE"] = "1"
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self._run(1, fake)
        self.assertEqual(fake.calls, 2, "NO_CACHE=1 应禁用复用，每轮都跑")

    def test_missing_prev_report_falls_back(self) -> None:
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self.assertEqual(fake.calls, 1)

        # 删掉上轮 report → 即便 hash 命中也应回退重跑。
        (self.task_dir / "precision_tuning" / "forensics_report_0.json").unlink()
        r1 = self._run(1, fake)
        self.assertEqual(fake.calls, 2, "上轮 report 缺失应回退重跑")
        self.assertFalse(r1.get("cached"))


class TestRunFullEval(unittest.TestCase):
    """全量复验 (§2.1): 备份→覆盖<op>.json→跑→finally恢复 的隔离正确性。

    mock _run_logged 不真跑 verification: 按需写 case 行到 stdout 控制 match_rate，
    核心断言全量跑前后生效 <op>.json 字节一致 (恢复正确，不污染 agent 后续输入)。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        (self.task_dir / "precision_tuning").mkdir(parents=True)
        # 轻量 5_FakeOp.json (2 行) + 全量 .bak (4 行)，含 inputs 键以过 _looks_like_case_jsonl。
        self.light = self.task_dir / "5_FakeOp.json"
        self.light.write_text(
            '{"inputs": [1]}\n{"inputs": [2]}\n', encoding="utf-8")
        self.full = self.task_dir / "5_FakeOp.json.bak"
        self.full.write_text(
            '{"inputs": [1]}\n{"inputs": [2]}\n{"inputs": [3]}\n{"inputs": [4]}\n',
            encoding="utf-8")
        self._env_backup = {k: os.environ.pop(k, None)
                            for k in ("ABLATE_FULL_EVAL", "ASCENDC_ABLATE_FULL_EVAL")}

    def tearDown(self) -> None:
        for k, v in self._env_backup.items():
            if v is not None:
                os.environ[k] = v
        self._tmp.cleanup()

    def _fake_run_logged(self, *, case_lines: str, rc: int):
        """返回一个 fake _run_logged: 把 case_lines 写进 stdout_path 并返回 rc。"""
        def _fake(cmd, *, cwd, env, stdout_path, stderr_path, title, timeout):
            Path(stdout_path).write_text(case_lines, encoding="utf-8")
            return rc
        return _fake

    def _call(self):
        return validate_runner._run_full_eval(
            self.task_dir, attempt=0, repo_root=Path("/repo"),
            env={}, clean_build=True, timeout=None)

    def test_full_eval_all_matched_restores_json(self) -> None:
        before = self.light.read_bytes()
        cases = "case[0]: output: matched\ncase[1]: output: matched\n" \
                "case[2]: output: matched\ncase[3]: output: matched\n"
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines=cases, rc=0)):
            fe = self._call()
        self.assertTrue(fe["ran"])
        self.assertFalse(fe["crashed"])
        self.assertEqual(fe["total_cases"], 4)
        self.assertEqual(fe["passed_cases"], 4)
        self.assertEqual(fe["full_json_source"], "5_FakeOp.json.bak")
        # 关键: 生效 <op>.json 恢复为轻量原文 (字节一致)。
        self.assertEqual(self.light.read_bytes(), before)
        # 备份文件已删。
        self.assertFalse(
            (self.task_dir / "5_FakeOp.json.lightweight_bak_attempt0").exists())
        # 单独落盘 + 跨轮状态。
        self.assertTrue(
            (self.task_dir / "precision_tuning" / "validation_result_attempt_0_full.json").exists())
        self.assertTrue(
            (self.task_dir / "precision_tuning" / ".full_eval_state.json").exists())

    def test_full_eval_partial_fail_restores_json(self) -> None:
        before = self.light.read_bytes()
        cases = "case[0]: output: matched\ncase[1]: output: matched\n" \
                "case[2]: output: mismatch_ratio=10.000000%\ncase[3]: output: matched\n"
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines=cases, rc=1)):
            fe = self._call()
        self.assertEqual(fe["total_cases"], 4)
        self.assertEqual(fe["passed_cases"], 3)
        self.assertFalse(fe["crashed"])
        self.assertEqual(self.light.read_bytes(), before)

    def test_full_eval_crash_restores_json(self) -> None:
        before = self.light.read_bytes()
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines="Traceback...\n", rc=1)):
            fe = self._call()
        self.assertTrue(fe["crashed"])  # rc!=0 且无 case 数据
        self.assertEqual(self.light.read_bytes(), before)

    def test_no_fuller_json_returns_none(self) -> None:
        # 删掉 .bak → 只剩轻量 → 无更全集合 → None (only_py 算子同此态)。
        self.full.unlink()
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines="", rc=0)):
            fe = self._call()
        self.assertIsNone(fe)


if __name__ == "__main__":
    unittest.main()
