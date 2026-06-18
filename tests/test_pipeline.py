from __future__ import annotations

from io import BytesIO
import json
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local3dai.agent import AgentRunOptions, run_agent_render
from local3dai.ai.backends import Flux2KleinBackend, MockImageBackend, resolve_flux_reference_channels, resolve_sdxl_control_channels
from local3dai.ai.geometry import (
    mesh_adaptive_lock_render,
    mesh_detail_lock_render,
    mesh_position_lock_render,
    mesh_quality_lock_render,
)
from local3dai.camera import CameraState, camera_position
from local3dai.modelgen import generate_3d_model
from local3dai.rendering.blender import render_model_with_blender
from local3dai.sample import create_sample_renders
from local3dai.scoring import score_candidates
from local3dai.scoring_v2 import score_structure_v2
from local3dai.webapp import (
    JOBS,
    JOBS_LOCK,
    StageError,
    _append_log,
    _init_stage_states,
    _job_snapshot,
    _read_multipart_upload,
    _run_hunyuan_shape_for_stage,
    _stage_ai_render,
    _stage_generate_3d,
    _stage_white_render,
)
from PIL import Image, ImageFilter


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

    def test_agent_default_model_key_uses_flux2_klein(self) -> None:
        self.assertEqual(AgentRunOptions(input_renders=Path("renders.json"), output_dir=Path("out"), prompt="x").model_key, "flux2_klein_4b")

    def test_flux2_reference_channel_resolution_supports_text_only_and_render_channels(self) -> None:
        self.assertEqual(resolve_flux_reference_channels({"reference_channels": []}), [])
        self.assertEqual(resolve_flux_reference_channels({"reference_channels": ["white", "canny", "depth", "normal", "silhouette", "bone"]}), ["rgb", "edge", "depth", "normal", "mask", "skeleton"])
        with self.assertRaises(RuntimeError):
            resolve_flux_reference_channels({"reference_channels": ["pose"]})

    def test_sdxl_control_channel_resolution_supports_canny_only_and_depth(self) -> None:
        self.assertEqual(resolve_sdxl_control_channels({"control_channels": ["canny"]}), ["canny"])
        self.assertEqual(resolve_sdxl_control_channels({"control_channels": ["canny", "depth"]}), ["canny", "depth"])
        with self.assertRaises(RuntimeError):
            resolve_sdxl_control_channels({"control_channels": ["normal"]})

    def test_flux2_backend_passes_multiple_reference_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_manifest = self._write_fake_render_manifest(root / "renders")
            fake_model = root / "flux2"
            fake_model.mkdir()
            appearance = root / "appearance.png"
            Image.new("RGB", (64, 64), (200, 20, 30)).save(appearance)
            detail = root / "detail.png"
            Image.new("RGB", (64, 64), (20, 200, 30)).save(detail)
            captured: list[dict[str, object]] = []

            class FakePipe:
                def to(self, device: str) -> None:
                    captured.append({"to": device})

                def __call__(
                    self,
                    *,
                    image: object = None,
                    prompt: str | None = None,
                    height: int | None = None,
                    width: int | None = None,
                    num_inference_steps: int | None = None,
                    guidance_scale: float | None = None,
                    generator: object = None,
                    max_sequence_length: int | None = None,
                ) -> object:
                    captured.append(
                        {
                            "image": image,
                            "prompt": prompt,
                            "height": height,
                            "width": width,
                            "steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "max_sequence_length": max_sequence_length,
                        }
                    )

                    class Result:
                        images = [Image.new("RGB", (64, 64), (40, 80, 140))]

                    return Result()

            with patch("diffusers.Flux2KleinPipeline.from_pretrained", return_value=FakePipe()):
                manifest = Flux2KleinBackend().generate(
                    render_manifest,
                    root / "flux_candidates",
                    prompt="clean render",
                    candidates_per_view=1,
                    seed=9,
                    model_ref=str(fake_model),
                    device="cpu",
                    dtype="bfloat16",
                    steps=4,
                    guidance_scale=1.0,
                    width=64,
                    height=64,
                    reference_channels=["rgb", "depth", "skeleton"],
                    appearance_reference=appearance,
                    appearance_reference_images=[detail],
                    appearance_reference_order="after",
                    detail_reference=detail,
                    mesh_position_lock=True,
                    geometry_lock=False,
                )
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["backend"], "flux2-klein")
            self.assertEqual(data["reference_channels"], ["rgb", "depth", "skeleton"])
            self.assertEqual(data["appearance_reference_order"], "after")
            self.assertTrue(data["mesh_position_lock"])
            self.assertEqual(len(data["appearance_reference_files"]), 2)
            self.assertEqual(data["detail_reference_file"], str(detail.resolve()))
            call = next(item for item in captured if "image" in item)
            self.assertIsInstance(call["image"], list)
            self.assertEqual(len(call["image"]), 5)
            self.assertEqual(call["image"][-2].getpixel((0, 0)), (200, 20, 30))
            self.assertEqual(call["image"][-1].getpixel((0, 0)), (20, 200, 30))
            refs = data["candidates"][0]["reference_files"]
            self.assertTrue(Path(refs["skeleton"]).exists())
            self.assertEqual(data["candidates"][0]["appearance_reference_files"], data["appearance_reference_files"])
            self.assertEqual(data["candidates"][0]["detail_reference_file"], str(detail.resolve()))

    def test_mesh_position_lock_clips_non_binary_mask_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            size = (96, 96)
            source_rgb = Image.new("RGB", size, (72, 74, 76))
            source_rgb.paste((230, 230, 232), (30, 30, 66, 66))
            source_mask = Image.new("L", size, 63)
            source_mask.paste(214, (30, 30, 66, 66))
            source_edge = source_mask.filter(ImageFilter.FIND_EDGES)
            ai = Image.new("RGB", size, (48, 50, 52))
            ai.paste((230, 20, 20), (30, 30, 66, 66))
            ai.paste((230, 20, 20), (70, 34, 90, 60))
            paths = {}
            for name, image in {
                "rgb": source_rgb,
                "mask": source_mask,
                "edge": source_edge,
                "ai": ai,
            }.items():
                path = root / f"{name}.png"
                image.save(path)
                paths[name] = path

            output = mesh_position_lock_render(
                source_rgb=paths["rgb"],
                source_mask=paths["mask"],
                source_edge=paths["edge"],
                ai_image=paths["ai"],
                output=root / "locked.png",
            )
            locked = Image.open(output).convert("RGB")
            inside = locked.getpixel((48, 48))
            outside_hallucination = locked.getpixel((80, 48))
            self.assertGreater(inside[0], 140)
            self.assertLess(outside_hallucination[0], 95)
            self.assertLess(outside_hallucination[0] - outside_hallucination[1], 15)

    def test_mesh_detail_lock_uses_mesh_edges_instead_of_shifted_ai_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            size = (96, 96)
            source_rgb = Image.new("RGB", size, (70, 72, 74))
            source_rgb.paste((220, 220, 222), (24, 24, 72, 72))
            source_mask = Image.new("L", size, 63)
            source_mask.paste(214, (24, 24, 72, 72))
            source_edge = Image.new("L", size, 80)
            for y in range(26, 70):
                source_edge.putpixel((48, y), 235)
            ai = Image.new("RGB", size, (48, 50, 52))
            ai.paste((230, 25, 20), (24, 24, 72, 72))
            for y in range(26, 70):
                ai.putpixel((66, y), (0, 0, 0))
                ai.putpixel((67, y), (0, 0, 0))
            detail = Image.new("RGB", size, (180, 20, 18))
            paths = {}
            for name, image in {
                "rgb": source_rgb,
                "mask": source_mask,
                "edge": source_edge,
                "ai": ai,
                "detail": detail,
            }.items():
                path = root / f"{name}.png"
                image.save(path)
                paths[name] = path

            output = mesh_detail_lock_render(
                source_rgb=paths["rgb"],
                source_mask=paths["mask"],
                source_edge=paths["edge"],
                ai_image=paths["ai"],
                output=root / "detail_locked.png",
                detail_reference=paths["detail"],
            )
            locked = Image.open(output).convert("RGB")
            shifted_ai_line = locked.getpixel((66, 48))
            mesh_edge_line = locked.getpixel((48, 48))
            self.assertGreater(shifted_ai_line[0], 80)
            self.assertGreater(shifted_ai_line[0], shifted_ai_line[1] * 2)
            self.assertLess(mesh_edge_line[0], shifted_ai_line[0])

    def test_mesh_adaptive_lock_uses_generic_edge_drift_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            size = (96, 96)
            source_rgb = Image.new("RGB", size, (70, 72, 74))
            source_rgb.paste((220, 220, 222), (24, 24, 72, 72))
            source_mask = Image.new("L", size, 63)
            source_mask.paste(214, (24, 24, 72, 72))
            source_edge = Image.new("L", size, 80)
            for y in range(26, 70):
                source_edge.putpixel((48, y), 235)
            ai = Image.new("RGB", size, (48, 50, 52))
            ai.paste((230, 25, 20), (24, 24, 72, 72))
            for y in range(26, 70):
                ai.putpixel((66, y), (0, 0, 0))
                ai.putpixel((67, y), (0, 0, 0))
            detail = Image.new("RGB", size, (180, 20, 18))
            paths = {}
            for name, image in {
                "rgb": source_rgb,
                "mask": source_mask,
                "edge": source_edge,
                "ai": ai,
                "detail": detail,
            }.items():
                path = root / f"{name}.png"
                image.save(path)
                paths[name] = path

            output = mesh_adaptive_lock_render(
                source_rgb=paths["rgb"],
                source_mask=paths["mask"],
                source_edge=paths["edge"],
                ai_image=paths["ai"],
                output=root / "adaptive_locked.png",
                detail_reference=paths["detail"],
            )
            locked = Image.open(output).convert("RGB")
            shifted_ai_line = locked.getpixel((66, 48))
            mesh_edge_line = locked.getpixel((48, 48))
            self.assertGreater(shifted_ai_line[0], mesh_edge_line[0])
            self.assertGreater(shifted_ai_line[0], shifted_ai_line[1] * 2)
            self.assertLess(mesh_edge_line[0], shifted_ai_line[0])

    def test_mesh_quality_lock_preserves_material_while_refining_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            size = (96, 96)
            source_rgb = Image.new("RGB", size, (70, 72, 74))
            source_rgb.paste((220, 220, 222), (24, 24, 72, 72))
            source_mask = Image.new("L", size, 63)
            source_mask.paste(214, (24, 24, 72, 72))
            source_edge = Image.new("L", size, 80)
            for y in range(26, 70):
                source_edge.putpixel((48, y), 235)
            ai = Image.new("RGB", size, (48, 50, 52))
            ai.paste((230, 25, 20), (24, 24, 72, 72))
            for y in range(26, 70):
                ai.putpixel((66, y), (0, 0, 0))
                ai.putpixel((67, y), (0, 0, 0))
            detail = Image.new("RGB", size, (180, 20, 18))
            paths = {}
            for name, image in {
                "rgb": source_rgb,
                "mask": source_mask,
                "edge": source_edge,
                "ai": ai,
                "detail": detail,
            }.items():
                path = root / f"{name}.png"
                image.save(path)
                paths[name] = path

            output = mesh_quality_lock_render(
                source_rgb=paths["rgb"],
                source_mask=paths["mask"],
                source_edge=paths["edge"],
                ai_image=paths["ai"],
                output=root / "quality_locked.png",
                detail_reference=paths["detail"],
            )
            locked = Image.open(output).convert("RGB")
            clean_body = locked.getpixel((36, 48))
            shifted_ai_line = locked.getpixel((66, 48))
            mesh_edge_line = locked.getpixel((48, 48))
            self.assertGreater(clean_body[0], 150)
            self.assertGreater(shifted_ai_line[0], 80)
            self.assertGreater(shifted_ai_line[0], shifted_ai_line[1] * 2)
            self.assertLess(mesh_edge_line[0], clean_body[0])

    def test_structure_score_v2_penalizes_shift_added_parts_and_background_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            size = (96, 96)
            source_mask = Image.new("L", size, 0)
            for x in range(28, 68):
                for y in range(26, 70):
                    source_mask.putpixel((x, y), 255)
            source_edge = source_mask.filter(ImageFilter.FIND_EDGES)
            source_rgb = Image.new("RGB", size, (245, 245, 245))
            source_rgb.paste((80, 80, 80), (28, 26, 68, 70))
            files = {}
            for name, image in {
                "rgb": source_rgb,
                "mask": source_mask.convert("RGB"),
                "edge": source_edge.convert("RGB"),
                "depth": source_mask.convert("RGB"),
                "normal": source_rgb,
            }.items():
                path = root / f"{name}.png"
                image.save(path)
                files[name] = str(path)

            def candidate(path: Path, box: tuple[int, int, int, int], *, extra: bool = False, noisy_bg: bool = False) -> Path:
                image = Image.new("RGB", size, (245, 245, 245))
                if noisy_bg:
                    for x in range(0, 96, 4):
                        for y in range(0, 96, 4):
                            image.putpixel((x, y), (40, 40, 40))
                image.paste((40, 70, 120), box)
                if extra:
                    image.paste((40, 70, 120), (72, 42, 88, 58))
                image.save(path)
                return path

            perfect = score_structure_v2(candidate_path=candidate(root / "perfect.png", (28, 26, 68, 70)), source_files=files)
            shifted = score_structure_v2(candidate_path=candidate(root / "shifted.png", (36, 26, 76, 70)), source_files=files)
            added = score_structure_v2(candidate_path=candidate(root / "added.png", (28, 26, 68, 70), extra=True), source_files=files)
            noisy = score_structure_v2(candidate_path=candidate(root / "noisy.png", (28, 26, 68, 70), noisy_bg=True), source_files=files)

            self.assertGreater(perfect["silhouette_iou"], shifted["silhouette_iou"])
            self.assertGreater(perfect["edge_chamfer_score"], shifted["edge_chamfer_score"])
            self.assertGreater(added["added_part_penalty"], perfect["added_part_penalty"])
            self.assertLess(noisy["background_cleanliness"], perfect["background_cleanliness"])

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
            self.assertTrue((root / "agent" / "multiview_contact_sheet.png").exists())
            self.assertEqual(summary["source_model_path"], "synthetic_sample")
            self.assertEqual(summary["reference_policy"], "model_render_channels_only")
            self.assertEqual(summary["model_key"], "flux2_klein_4b")
            self.assertEqual(summary["backend"], "mock")
            self.assertEqual(summary["score_version"], "structure_v2")
            self.assertEqual(summary["render_manifest"], str(render_manifest))
            self.assertGreaterEqual(len(summary["expanded_views"]), 2)
            self.assertIn("structure_scores", summary)
            self.assertIn("multiview_scores", summary)
            self.assertIn("retry_decisions", summary)
            channels = [trial["reference_channels"] for trial in summary["trials"]]
            self.assertTrue(all(channel_set == ["rgb", "edge", "depth", "normal", "mask", "skeleton"] for channel_set in channels))
            expanded_trial_ids = {item["trial_id"] for item in summary["expanded_views"]}
            self.assertIn(summary["selected_trial"]["trial_id"], expanded_trial_ids)

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
            self.assertEqual(len(summary["expanded_views"]), 3)
            self.assertTrue((root / "agent" / "three_view_contact.png").exists())
            self.assertTrue((root / "agent" / "multiview_contact_sheet.png").exists())

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

    def test_multipart_upload_parser_reads_file_field_without_cgi(self) -> None:
        boundary = "----harmonize3d-test"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="reference image.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + b"fake-png-bytes" + f"\r\n--{boundary}--\r\n".encode("utf-8")

        class Handler:
            headers = {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            }
            rfile = BytesIO(body)

        upload = _read_multipart_upload(Handler())  # type: ignore[arg-type]
        self.assertEqual(upload, ("reference image.png", b"fake-png-bytes"))

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

    def test_stage_ai_render_agent_defaults_to_flux2_model_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_manifest = create_sample_renders(root / "renders", views=3, resolution=96)
            captured: dict[str, object] = {}

            def fake_agent(options: AgentRunOptions) -> dict[str, object]:
                captured["model_key"] = options.model_key
                captured["backend"] = options.backend
                return {
                    "status": "complete",
                    "final_image": str(root / "final.png"),
                    "comparison_image": str(root / "comparison.png"),
                    "agent_report": str(root / "agent_report.json"),
                    "multiview_contact_sheet": str(root / "multiview_contact_sheet.png"),
                    "final_view_images": {"view_locked": str(root / "final_view_locked.png")},
                }

            with patch("local3dai.webapp.run_agent_render", side_effect=fake_agent):
                result = _stage_ai_render(
                    {
                        "render_manifest": str(render_manifest),
                        "prompt": "clean studio render",
                        "agent_render": True,
                        "backend": "mock",
                        "seed": 12,
                    }
                )
            self.assertEqual(captured["model_key"], "flux2_klein_4b")
            self.assertEqual(captured["backend"], "mock")
            self.assertEqual(result["summary"]["final_view_images"]["view_locked"], str(root / "final_view_locked.png"))

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
