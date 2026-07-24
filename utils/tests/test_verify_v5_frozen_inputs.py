from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "verify_v5_frozen_inputs.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_frozen_inputs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5FrozenInputTests(unittest.TestCase):
    def test_code_manifest_excludes_secret_output_and_git(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".secrets").mkdir()
            (root / "outputs").mkdir()
            (root / ".git").mkdir()
            (root / "utils").mkdir()
            (root / ".secrets" / "key.json").write_text("secret")
            (root / "outputs" / "result.json").write_text("result")
            (root / ".git" / "config").write_text("git")
            (root / "utils" / "tool.py").write_text("print('ok')\n")

            manifest = module.file_manifest(
                root, skip_runtime=True, code_root=True)

            self.assertEqual(list(manifest), ["utils/tool.py"])

    def test_compare_manifest_reports_changed_missing_and_extra(self) -> None:
        module = _load()
        result = module.compare_manifest(
            {"a": "1", "b": "2"},
            {"a": "9", "c": "3"},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["changed"], ["a"])
        self.assertEqual(result["missing"], ["b"])
        self.assertEqual(result["extra"], ["c"])

    def test_control_plane_manifest_detects_deployed_drift(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            deployed = root / "deployed"
            source.mkdir()
            deployed.mkdir()
            (source / "launch.sh").write_text("echo frozen\n")
            (deployed / "launch.sh").write_text("echo changed\n")

            source_manifest = module.selected_file_manifest(
                source, ["launch.sh"])
            deployed_manifest = module.selected_file_manifest(
                deployed, ["launch.sh"])
            result = module.compare_manifest(
                source_manifest, deployed_manifest)

            self.assertFalse(result["passed"])
            self.assertEqual(result["changed"], ["launch.sh"])


if __name__ == "__main__":
    unittest.main()
