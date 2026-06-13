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

    def __call__(self, cmd, cwd=None, env=None, capture_output=False,
                 text=False, timeout=None, check=False):
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


if __name__ == "__main__":
    unittest.main()
