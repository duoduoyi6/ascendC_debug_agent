from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = REPO_ROOT / "experiments" / "v5_ablation"
DESIGN = REPO_ROOT / "docs" / "v5_ablation_experiment_master_design_20260724.md"
ASSETS_ROOT = "/home/wsx/AscendOpGenAgent_assets"
OLD_PATHS = (
    str(Path("/root") / "cannbot_debug_inputs_n27_20260723"),
    str(Path("/root") / "v5_ablation_control_20260724"),
)


class V5AssetPathTests(unittest.TestCase):
    def test_v5_files_use_sibling_asset_root(self) -> None:
        paths = [
            path for path in V5_ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        paths.append(DESIGN)
        payload = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in paths
        )

        self.assertIn(ASSETS_ROOT, payload)
        for old_path in OLD_PATHS:
            self.assertNotIn(old_path, payload)


if __name__ == "__main__":
    unittest.main()
