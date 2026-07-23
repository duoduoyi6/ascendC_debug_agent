from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_v5_posthoc_observer.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_posthoc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5PosthocIsolationTests(unittest.TestCase):
    def test_prepare_uses_full_cases_without_mutating_treatment(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "level1" / "001_Foo"
            treatment = root / "arm" / "tasks" / "level1" / "001_Foo"
            work = root / "posthoc" / "work" / "level1" / "001_Foo"
            (source / "kernel").mkdir(parents=True)
            (treatment / "kernel" / "build").mkdir(parents=True)
            (treatment / ".verify_logs").mkdir()
            (source / "model.py").write_text("# frozen\n", encoding="utf-8")
            (source / "model_new_ascendc.py").write_text(
                "# original candidate\n", encoding="utf-8")
            (source / "1_Foo.json").write_text(
                '{"inputs":[1]}\n{"inputs":[2]}\n', encoding="utf-8")
            (source / "1_Foo.json.bak").write_text(
                '{"inputs":[1]}\n{"inputs":[99]}\n', encoding="utf-8")
            (treatment / "model.py").write_text("# treatment\n", encoding="utf-8")
            (treatment / "model_new_ascendc.py").write_text(
                "# candidate\n", encoding="utf-8")
            (treatment / "1_Foo.json").write_text(
                '{"inputs":[1]}\n', encoding="utf-8")
            (treatment / "kernel" / "foo.cpp").write_text(
                "// candidate\n", encoding="utf-8")
            (treatment / "kernel" / "build" / "stale.o").write_text(
                "stale\n", encoding="utf-8")
            (treatment / ".verify_logs" / "old.stdout").write_text(
                "old\n", encoding="utf-8")

            target = module.Target(
                rel="level1/001_Foo", treatment=treatment, source=source)
            result = module.prepare_isolated_task(target, work)

            self.assertTrue(result["available"])
            self.assertFalse(result["coverage_equivalent"])
            self.assertEqual(
                (work / "1_Foo.json").read_text(encoding="utf-8"),
                (source / "1_Foo.json.bak").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (treatment / "1_Foo.json").read_text(encoding="utf-8"),
                '{"inputs":[1]}\n',
            )
            self.assertEqual(
                (work / "model.py").read_text(encoding="utf-8"), "# frozen\n")
            self.assertFalse((work / "kernel" / "build").exists())
            self.assertFalse((work / ".verify_logs").exists())
            self.assertEqual(
                (work / ".bench_baseline" / "model_new_ascendc.py").read_text(),
                "# original candidate\n",
            )
            self.assertNotEqual(
                (work / ".bench_baseline" / "model_new_ascendc.py").read_text(),
                (work / "model_new_ascendc.py").read_text(),
            )

    def test_case_digest_ignores_formatting_and_order(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.json"
            second = root / "b.json"
            first.write_text(
                '{"inputs":[1],"meta":{"b":2,"a":1}}\n{"inputs":[2]}\n',
                encoding="utf-8",
            )
            second.write_text(
                '{"inputs": [2]}\n{"meta":{"a":1,"b":2},"inputs":[1]}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                module._case_set_digest(first),
                module._case_set_digest(second),
            )


if __name__ == "__main__":
    unittest.main()
