from __future__ import annotations

import json
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local3dai.agent import AgentRunOptions, run_agent_render
from local3dai.ai.backends import MockImageBackend
from local3dai.camera import CameraState, camera_position
from local3dai.modelgen import generate_3d_model
from local3dai.rendering.blender import render_model_with_blender
from local3dai.sample import create_sample_renders
from local3dai.scoring import score_candidates
from local3dai.webapp import (
    JOBS,
    JOBS_LOCK,
    StageError,
    _append_log,
    _init_stage_states,
    _job_snapshot,
    _run_hunyuan_shape_for_stage,
    _stage_ai_render,
    _stage_generate_3d,
    _stage_white_render,
)
from PIL import Image


class PipelineTest(unittest.TestCase):
    def _write_fake_render_manifest(self, output: Path, *, view_id: str = "view_locked") -> Path:
        view_dir = output / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        for channel in ("rgb", "depth", "edge", "normal", "mask"):
            path = view_dir / f"{channel}.png"
            color = (240, 242, 244) if channel != "mask" else (255, 255, 255)
            image = Image.new("RGB", (32, 32), color)
            if channel in {"rgb", "normal"}:
                image.putpixel((0, 0), (8, 10, 12))
            image.save(path)
            files[channel] = str(path)
        manifest = {"type": "render_manifest", "source": "test", "views": [{"view_id": view_id, "camera": {}, "files": files}]}
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_camera_state_converts_to_blender_orbit_position(self) -> None:
        state = CameraState(azimuth_deg=0, elevation_deg=0, distance_scale=1.0, ortho_scale=2.7, target=(0.0, 0.0, 0.05))
        x, y, z = camera_position(state, base_distance=3.2)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, -3.2, places=5)
        self.assertAlmostEqual(z, 0.05, places=5)

    def test_three_y_up_camera_state_converts_to_blender_z_up(self) -> None:
        state = CameraState(
            azimuth_deg=35,
            elevation_deg=18,
            distance_scale=1.0,
            ortho_scale=2.7,
            target=(0.0, 0.05, 0.0),
            position=(1.7, 1.05, 2.4),
            coordinate_space="three_y_up",
        )
        converted = state.to_blender(model_path="model.glb")
        self.assertEqual(converted.coordinate_space, "blender_z_up")
        self.assertEqual(converted.target, (0.0, -0.0, 0.05))
        self.assertEqual(converted.position, (1.7, -2.4, 1.05))
        self.assertEqual(camera_position(converted), (1.7, -2.4, 1.05))

    def test_blender_wrapper_passes_camera_json_for_locked_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.obj"
            model.write_text("o test\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

            def fake_run(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                self.assertIn("--camera-json", command)
                output = Path(command[command.index("--output") + 1])
                self._write_fake_render_manifest(output, view_id="view_locked")

            with patch("local3dai.rendering.blender.subprocess.run", side_effect=fake_run):
                manifest_path = render_model_with_blender(
                    model_path=model,
                    output_dir=root / "renders",
                    blender_script=root / "batch_render.py",
                    blender_path="/bin/true",
                    camera=CameraState().to_dict(),
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["views"][0]["view_id"], "view_locked")

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

    def test_agent_render_mock_budget_outputs_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_manifest = create_sample_renders(root / "renders", views=4, resolution=128)
            summary = run_agent_render(
                AgentRunOptions(
                    input_renders=render_manifest,
                    output_dir=root / "agent",
                    prompt="premium red concept sports car, product render",
                    config_path=Path("configs/local.json"),
                    backend="mock",
                    max_generations=4,
                    seed=123,
                    pass_threshold=0.0,
                )
            )
            self.assertLessEqual(len(summary["trials"]), 4)
            self.assertTrue((root / "agent" / "agent_report.json").exists())
            self.assertTrue((root / "agent" / "final.png").exists())
            self.assertTrue((root / "agent" / "white_vs_final.png").exists())
            self.assertTrue((root / "agent" / "three_view_contact.png").exists())
            self.assertEqual(summary["source_model_path"], "synthetic_sample")
            self.assertEqual(summary["reference_policy"], "model_render_channels_only")
            self.assertEqual(summary["render_manifest"], str(render_manifest))
            self.assertGreaterEqual(len(summary["expanded_views"]), 2)
            channels = [trial["reference_channels"] for trial in summary["trials"]]
            self.assertTrue(all(channel_set == ["rgb", "edge"] for channel_set in channels))
            selected_score = summary["selected_trial"]["scores"]["total"]
            best_score = max(trial["scores"]["total"] for trial in summary["trials"] if trial["view_id"] == summary["target_view"])
            self.assertEqual(selected_score, best_score)

    def test_agent_render_single_view_when_threshold_not_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_manifest = create_sample_renders(root / "renders", views=3, resolution=96)
            summary = run_agent_render(
                AgentRunOptions(
                    input_renders=render_manifest,
                    output_dir=root / "agent",
                    prompt="clean product render",
                    config_path=Path("configs/local.json"),
                    backend="mock",
                    max_generations=3,
                    seed=456,
                    pass_threshold=1.1,
                )
            )
            self.assertEqual(summary["status"], "needs_review")
            self.assertEqual(len(summary["expanded_views"]), 1)
            self.assertFalse((root / "agent" / "three_view_contact.png").exists())

    def test_web_job_tracks_independent_stages(self) -> None:
        job_id = "stage-test"
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "status": "running",
                "stage": "queued",
                "progress": 0.0,
                "stages": _init_stage_states(agent_render=True),
                "logs": [],
            }
        try:
            _append_log(job_id, "prepare", "Preparing", 0.1)
            _append_log(job_id, "source", "Resolving 3D source", 0.2)
            snapshot = _job_snapshot(job_id)
            stages = {stage["id"]: stage for stage in snapshot["stages"]}
            self.assertEqual(stages["prepare"]["status"], "complete")
            self.assertEqual(stages["source"]["status"], "running")
            self.assertIn("agent", stages)
            self.assertNotIn("ai", stages)
            self.assertNotIn("score", stages)
            self.assertEqual(stages["source"]["message"], "Resolving 3D source")
        finally:
            with JOBS_LOCK:
                JOBS.pop(job_id, None)

    def test_stage_generate_3d_existing_model_returns_web_url(self) -> None:
        result = _stage_generate_3d(
            {
                "source_mode": "model_path",
                "model_path": str(Path("examples/sample_model.obj").resolve()),
                "seed": 1,
            }
        )
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["model_path"].endswith("sample_model.obj"))
        self.assertIn("/api/file?path=", result["model_url"])

    def test_stage_hunyuan_oom_error_keeps_reference_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.png"
            Image.new("RGB", (16, 16), (255, 255, 255)).save(reference)

            with patch(
                "local3dai.webapp._run_hunyuan_shape",
                side_effect=subprocess.CalledProcessError(-9, ["hunyuan"]),
            ):
                with self.assertRaises(StageError) as raised:
                    _run_hunyuan_shape_for_stage(
                        config={"models": {"hunyuan3d_2_1_shape": {}}},
                        reference_image=reference,
                        output_model=root / "white_mesh.glb",
                        metadata=root / "metadata.json",
                        seed=1,
                        source_mode="prompt_3d",
                        workdir=root,
                        profiles=[("stable", {"steps": 5, "octree_resolution": 256, "num_chunks": 8000})],
                    )

            payload = raised.exception.payload
            self.assertEqual(payload["error_type"], "hunyuan_oom")
            self.assertEqual(payload["reference_image"], str(reference))
            self.assertIn("/api/file?path=", payload["reference_image_url"])
            self.assertIn("available_plus_swap_mib", payload["memory"])

    def test_stage_generate_3d_returns_hunyuan_metadata_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.png"
            Image.new("RGB", (32, 32), (245, 245, 245)).save(reference)

            def fake_run_hunyuan_shape_for_stage(**kwargs: object) -> Path:
                output_model = Path(kwargs["output_model"])
                metadata = Path(kwargs["metadata"])
                preprocessed = output_model.with_name("shape_input_preprocessed.png")
                output_model.write_text("glb", encoding="utf-8")
                Image.new("RGBA", (32, 32), (255, 255, 255, 0)).save(preprocessed)
                metadata.write_text(
                    json.dumps(
                        {
                            "source_image": str(reference),
                            "preprocessed_image": str(preprocessed),
                            "mesh_sanity": {"status": "pass", "vertices": 50000, "faces": 100000, "flags": []},
                        }
                    ),
                    encoding="utf-8",
                )
                return output_model

            with patch("local3dai.webapp._run_hunyuan_shape_for_stage", side_effect=fake_run_hunyuan_shape_for_stage):
                result = _stage_generate_3d(
                    {
                        "source_mode": "image_3d",
                        "reference_image": str(reference),
                        "shape_quality": "balanced",
                        "seed": 1,
                    }
                )
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["preprocessed_image"].endswith("shape_input_preprocessed.png"))
            self.assertEqual(result["mesh_sanity"]["status"], "pass")

    def test_hunyuan_preprocess_writes_shape_input(self) -> None:
        script_path = Path("scripts/run_hunyuan_shape_white.py").resolve()
        spec = importlib.util.spec_from_file_location("run_hunyuan_shape_white", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.png"
            image = Image.new("RGBA", (80, 48), (255, 255, 255, 0))
            for x in range(20, 60):
                for y in range(12, 36):
                    image.putpixel((x, y), (240, 240, 240, 255))
            image.save(reference)
            output = root / "shape_input_preprocessed.png"
            metadata = module.preprocess_reference_image(
                reference,
                output,
                repo_root=Path("tools/Hunyuan3D-2.1").resolve(),
                remove_background=False,
            )
            self.assertTrue(output.exists())
            with Image.open(output) as processed:
                self.assertEqual(processed.mode, "RGBA")
                self.assertEqual(processed.width, processed.height)
            self.assertFalse(metadata["used_background_remover"])
            self.assertGreater(metadata["foreground_ratio"], 0)

    def test_stage_white_render_returns_locked_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_render_model_with_blender(**kwargs: object) -> Path:
                self.assertEqual(kwargs["camera"]["azimuth_deg"], 35.0)
                return self._write_fake_render_manifest(Path(kwargs["output_dir"]), view_id="view_locked")

            with patch("local3dai.webapp.render_model_with_blender", side_effect=fake_render_model_with_blender):
                result = _stage_white_render(
                    {
                        "model_path": str(Path("examples/sample_model.obj").resolve()),
                        "camera": CameraState().to_dict(),
                        "resolution": 64,
                        "samples": 8,
                    }
                )
            manifest = json.loads(Path(result["render_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["views"][0]["view_id"], "view_locked")
            self.assertIn("rgb", result["channel_urls"])

    def test_stage_white_render_converts_three_camera_to_blender(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_render_model_with_blender(**kwargs: object) -> Path:
                camera = kwargs["camera"]
                self.assertEqual(camera["coordinate_space"], "blender_z_up")
                self.assertEqual(camera["target"], [0.0, -0.0, 0.05])
                self.assertEqual(camera["position"], [1.7, -2.4, 1.05])
                return self._write_fake_render_manifest(Path(kwargs["output_dir"]), view_id="view_locked")

            with patch("local3dai.webapp.render_model_with_blender", side_effect=fake_render_model_with_blender):
                result = _stage_white_render(
                    {
                        "model_path": str(Path("examples/sample_model.obj").resolve()),
                        "camera": {
                            "azimuth_deg": 35,
                            "elevation_deg": 18,
                            "distance_scale": 1.0,
                            "ortho_scale": 2.7,
                            "target": [0.0, 0.05, 0.0],
                            "position": [1.7, 1.05, 2.4],
                            "viewport_aspect": 1.0,
                            "coordinate_space": "three_y_up",
                        },
                        "resolution": 64,
                        "samples": 8,
                    }
                )
            self.assertEqual(result["camera"]["coordinate_space"], "three_y_up")
            self.assertEqual(result["blender_camera"]["coordinate_space"], "blender_z_up")

    def test_stage_ai_render_mock_outputs_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_manifest = create_sample_renders(root / "renders", views=1, resolution=96)
            result = _stage_ai_render(
                {
                    "render_manifest": str(render_manifest),
                    "prompt": "clean studio render",
                    "agent_render": False,
                    "backend": "mock",
                    "seed": 12,
                }
            )
            self.assertEqual(result["status"], "complete")
            self.assertTrue(Path(result["final_image"]).exists())
            self.assertTrue(Path(result["comparison_image"]).exists())

    def test_stage_ai_render_rejects_raw_reference_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_manifest = create_sample_renders(root / "renders", views=1, resolution=96)
            with self.assertRaises(StageError) as raised:
                _stage_ai_render(
                    {
                        "render_manifest": str(render_manifest),
                        "reference_image": str(root / "reference.png"),
                        "prompt": "clean studio render",
                    }
                )
            self.assertEqual(raised.exception.payload["stage"], "ai-render")
            self.assertIn("reference_image", raised.exception.payload["forbidden_fields"])


if __name__ == "__main__":
    unittest.main()
