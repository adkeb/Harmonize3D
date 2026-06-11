from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local3dai.ai.backends import MockImageBackend
from local3dai.modelgen import generate_3d_model
from local3dai.sample import create_sample_renders
from local3dai.scoring import score_candidates


class PipelineTest(unittest.TestCase):
    def test_mock_pipeline_scores_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_manifest = create_sample_renders(root / "renders", views=2, resolution=128)
            ai_manifest = MockImageBackend().generate(
                render_manifest,
                root / "candidates",
                prompt="studio render",
                candidates_per_view=2,
                seed=123,
            )
            report_path = score_candidates(
                render_manifest_path=render_manifest,
                ai_manifest_path=ai_manifest,
                output_dir=root / "score",
                copy_top_k=1,
            )
            with report_path.open("r", encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertEqual(report["count"], 4)
            self.assertTrue((root / "score" / "ranked").exists())
            self.assertGreaterEqual(report["ranked"][0]["scores"]["total"], report["ranked"][-1]["scores"]["total"])

    def test_sample_model_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generated.obj"
            model_path = generate_3d_model(prompt="test object", output=output, backend="sample")
            self.assertEqual(model_path, output)
            self.assertTrue(model_path.exists())
            self.assertGreater(model_path.stat().st_size, 0)

    def test_procedural_crystal_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "crystal.obj"
            model_path = generate_3d_model(prompt="neon crystal reactor", output=output, backend="procedural-crystal")
            self.assertEqual(model_path, output)
            self.assertTrue(model_path.exists())
            self.assertTrue(output.with_suffix(".mtl").exists())
            text = model_path.read_text(encoding="utf-8")
            self.assertIn("faceted_neon_core", text)
            self.assertIn("tilted_energy_halo", text)


if __name__ == "__main__":
    unittest.main()
