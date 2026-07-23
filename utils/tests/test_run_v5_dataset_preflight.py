from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_v5_dataset_preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_dataset_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5DatasetPreflightTests(unittest.TestCase):
    def test_prepare_is_deep_copy_and_removes_runtime(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "level1" / "001_Foo"
            work = root / "preflight" / "work" / "level1" / "001_Foo"
            (source / "kernel" / "build").mkdir(parents=True)
            (source / ".verify_logs").mkdir()
            (source / "model.py").write_text("# source\n")
            (source / "kernel" / "foo.cpp").write_text("// kernel\n")
            (source / "kernel" / "build" / "stale.o").write_text("stale\n")
            (source / ".verify_logs" / "old.log").write_text("old\n")
            target = module.Target(rel="level1/001_Foo", source=source)

            module.prepare_isolated_task(target, work)

            self.assertTrue((work / "model.py").is_file())
            self.assertTrue((work / "kernel" / "foo.cpp").is_file())
            self.assertFalse((work / "kernel" / "build").exists())
            self.assertFalse((work / ".verify_logs").exists())
            (work / "model.py").write_text("# changed\n")
            self.assertEqual((source / "model.py").read_text(), "# source\n")

    def test_debuggable_failure_types_are_closed(self) -> None:
        module = _load()
        self.assertEqual(
            module.DEBUGGABLE_FAILURE_TYPES,
            {
                "build_failed", "import_failed", "runtime_error",
                "timeout", "precision_failed",
            },
        )


if __name__ == "__main__":
    unittest.main()
