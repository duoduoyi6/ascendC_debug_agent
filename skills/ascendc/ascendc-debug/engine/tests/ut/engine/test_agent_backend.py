"""test_agent_backend.py — diagnose_and_fix 的 claude backend (方案 C 唯一 spawn 点)。

不真跑 claude: 用 fake _run 注入，验证三件事:
  1. Action → claude 命令行参数正确 (--agent/--session-id/--add-dir/--allowedTools/prompt)。
  2. claude --output-format json 结果 → result dict 映射 (ok/pause_turn/error/api_error)。
  3. make_agent_callback 闭包注入 runner，跑通 precision→CONTINUE→PASS 2 轮 (claude 被 fake，
     gate 用 mock dispatcher)，确认 backend 与 runner 接口契合。
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from engine.agent_backend import (
    make_agent_callback,
    spawn_diagnose_agent,
)
from engine.types import Action


def _diagnose_action(failure_type="precision_failed", attempt=0) -> Action:
    return Action(kind="spawn_agent", name="debug_worker", step="diagnose_and_fix",
                  skill_args={"failure_type": failure_type, "attempt": attempt})


def _fake_run_factory(claude_json: dict, capture: dict):
    """造一个 fake subprocess.run: 记录 cmd，并把 claude_json 写进 stdout 文件句柄。"""
    def _fake_run(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
        capture["cmd"] = cmd
        capture["timeout"] = timeout
        if stdout is not None:
            stdout.write(json.dumps(claude_json))
        class _R:
            returncode = 0
        return _R()
    return _fake_run


class TestCommandConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_cmd_has_required_flags(self) -> None:
        cap = {}
        spawn_diagnose_agent(
            _diagnose_action("build_failed", 2), self.task_dir, "add", 2,
            agent_name="ascendc-debug-agent-constructive", model="claude-x",
            allowed_tools="Bash,Read", npu="3",
            _run=_fake_run_factory({"is_error": False}, cap))
        cmd = cap["cmd"]
        self.assertIn("--agent", cmd)
        self.assertEqual(cmd[cmd.index("--agent") + 1], "ascendc-debug-agent-constructive")
        self.assertIn("--session-id", cmd)
        self.assertIn("--add-dir", cmd)
        self.assertIn("--allowedTools", cmd)
        self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "Bash,Read")
        self.assertIn("--output-format", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
        # prompt 是最后一个位置参数，含本轮 failure_type + attempt + 单轮约束。
        prompt = cmd[-1]
        self.assertIn("build_failed", prompt)
        self.assertIn("attempt: 2", prompt)
        self.assertIn("单个 attempt", prompt)
        self.assertIn("禁止自己跑", prompt)

# PLACEHOLDER_TESTS


class TestResultClassification(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _spawn(self, claude_json):
        cap = {}
        return spawn_diagnose_agent(
            _diagnose_action(), self.task_dir, "add", 0,
            _run=_fake_run_factory(claude_json, cap))

    def test_ok_result_success(self) -> None:
        r = self._spawn({"is_error": False, "stop_reason": "end_turn"})
        self.assertTrue(r["success"])
        self.assertEqual(r["claude_state"], "ok")
        self.assertIn("session_id", r)

    def test_pause_turn_not_success(self) -> None:
        # pause_turn = 本轮未自然结束，不算正常完成 (runner 据此不当作干净一轮)。
        r = self._spawn({"is_error": False, "stop_reason": "pause_turn"})
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "claude_pause_turn")

    def test_is_error_not_success(self) -> None:
        r = self._spawn({"is_error": True, "result": "boom"})
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "claude_error")

    def test_api_error_not_success(self) -> None:
        r = self._spawn({"is_error": True, "api_error_status": 401})
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "api_error_401")

    def test_empty_result_file_invalid(self) -> None:
        # 真实流程: spawn 用 open("w") 必先建文件，fake _run 不写内容 → 空文件 →
        # json 解析失败 → invalid_claude_result (而非 missing；missing 仅在 open 都未发生)。
        def _no_write(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
            class _R:
                returncode = 0
            return _R()
        r = spawn_diagnose_agent(_diagnose_action(), self.task_dir, "add", 0, _run=_no_write)
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "invalid_claude_result")

    def test_spawn_exception_fatal(self) -> None:
        def _boom(*a, **k):
            raise OSError("claude not found")
        r = spawn_diagnose_agent(_diagnose_action(), self.task_dir, "add", 0, _run=_boom)
        self.assertFalse(r["success"])
        self.assertTrue(r["fatal"])
        self.assertEqual(r["claude_state"], "spawn_failed")

    def test_timeout(self) -> None:
        def _slow(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
            raise subprocess.TimeoutExpired(cmd, timeout)
        r = spawn_diagnose_agent(_diagnose_action(), self.task_dir, "add", 0,
                                 timeout_sec=1, _run=_slow)
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "timeout")

# PLACEHOLDER_INTEGRATION


class TestCallbackIntoRunner(unittest.TestCase):
    """make_agent_callback 注入 runner，跑通 precision→CONTINUE→PASS 2 轮。

    claude 被 fake (不真跑)；gate 用自定义 dispatcher 喂预设 loop_signal。验证 backend
    的 callback 在 runner 真实调用链里被正确触发，且 session 抵达 success。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_callback_invoked_session_to_success(self) -> None:
        from engine.runner import run_debug_session

        spawn_calls = []

        def _fake_run(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
            spawn_calls.append(cmd[cmd.index("--agent") + 1])
            if stdout is not None:
                stdout.write(json.dumps({"is_error": False, "stop_reason": "end_turn"}))
            class _R:
                returncode = 0
            return _R()

        callback = make_agent_callback(
            agent_name="ascendc-debug-agent-constructive", _run=_fake_run)

        # 自定义 dispatcher: spawn_agent 走 backend callback；py_action 喂预设 gate。
        validate_seq = iter([
            {"gate": "GATE-V", "passed": False, "loop_signal": "CONTINUE"},
            {"gate": "GATE-V", "passed": True, "loop_signal": "PASS"},
        ])

        def _dispatcher(action, task_dir, op_name, agent_callback):
            if action.kind == "spawn_agent":
                return agent_callback(action, task_dir, op_name,
                                      (action.skill_args or {}).get("attempt", 0))
            if action.step == "validate":
                return dict(next(validate_seq))
            return {"success": True}

        status = run_debug_session(
            self.task_dir, op_name="add", agent="constructive",
            entry_failure_type="precision_failed",
            agent_callback=callback, dispatcher=_dispatcher)

        self.assertEqual(status["session_outcome"], "success")
        self.assertEqual(status["attempts_used"], 2)
        # backend callback 被触发 2 次 (每轮 diagnose_and_fix 一次)，都用 constructive。
        self.assertEqual(spawn_calls,
                         ["ascendc-debug-agent-constructive"] * 2)


if __name__ == "__main__":
    unittest.main()


