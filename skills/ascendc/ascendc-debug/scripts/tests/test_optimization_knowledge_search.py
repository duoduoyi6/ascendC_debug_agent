from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "optimization_knowledge.py"
KB = Path(__file__).resolve().parents[2] / "references" / "optimization_knowledge_base.json"


class OptimizationKnowledgeTest(unittest.TestCase):
    def test_validate_canonical_kb(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "validate", "--kb-path", str(KB)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["entries"], 13)
        self.assertGreaterEqual(payload["promoted"], 1)

    def test_canonical_kb_is_hardware_agnostic(self):
        entries = json.loads(KB.read_text(encoding="utf-8"))
        forbidden = {"hardware", "hardware_scope", "device_model", "soc", "soc_version"}
        for entry in entries:
            self.assertFalse(forbidden.intersection(entry))
        self.assertIsNone(re.search(r"\b910\s*[bc]\b", KB.read_text(encoding="utf-8"), re.I))

    def test_fine_search_prefers_matching_bottleneck_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "log.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "search",
                    "--kb-path",
                    str(KB),
                    "--op-name",
                    "RmsNorm",
                    "--op-type",
                    "normalization",
                    "--pattern",
                    "redundant_gm_pass",
                    "--bottleneck",
                    "ub_underutilization",
                    "--phase",
                    "fine",
                    "--log-path",
                    str(log),
                    "--top-k",
                    "2",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["matches"][0]["knowledge"]["type"], "OPT_UB_RESIDENT_FAST_PATH")
            self.assertTrue(log.exists())

    def test_ablation_guard(self):
        env = dict(os.environ)
        env["ABLATE_OPTIMIZATION_KB"] = "1"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "validate", "--kb-path", str(KB)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(json.loads(proc.stdout)["reason"], "ABLATE_OPTIMIZATION_KB=1")


if __name__ == "__main__":
    unittest.main()
