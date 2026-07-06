from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local3dai.cli import build_parser
from local3dai.manifest import read_manifest, write_manifest
from PIL import Image, ImageDraw


class AutoSceneCliTest(unittest.TestCase):
    def _write_retry_workdir(self, root: Path) -> Path:
        retry_request = root / "final" / "codex_image2_position_retry_request.json"
        retry_request.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(
            retry_request,
            {
                "type": "codex_image2_position_retry_request",
                "status": "awaiting_codex_image2",
                "provider": "codex_builtin_image2",
                "requests": [
                    {
                        "view_id": "view_hero",
                        "kind": "final_render_position_retry",
                        "output_path": str(root / "final" / "final_view_hero.png"),
                    }
                ],
            },
        )
        write_manifest(
            root / "reports" / "final_position_retry_plan.json",
            {
                "type": "final_position_retry_plan",
                "status": "awaiting_codex_image2_retry",
                "retry_request": str(retry_request),
                "codex_image2_handoff": str(root / "final" / "codex_image2_position_retry_handoff.md"),
            },
        )
        return retry_request

    def _run_cli(self, argv: list[str]) -> tuple[int, dict]:
        parser = build_parser()
        args = parser.parse_args(argv)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = args.func(args)
        return code, json.loads(output.getvalue())

    def test_auto_scene_position_retry_dry_plan_prints_handoff_without_import_or_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_retry_workdir(root)

            with patch("local3dai.cli.import_latest_codex_image2_results") as import_latest, patch("local3dai.cli.run_auto_scene") as run_scene:
                code, result = self._run_cli(["auto-scene-run-position-retry", "--workdir", str(root), "--dry-plan"])

            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "plan_only")
            self.assertEqual(result["retry_plan_status"], "awaiting_codex_image2_retry")
            self.assertIn("codex_image2_position_retry_request.json", result["retry_request"])
            import_latest.assert_not_called()
            run_scene.assert_not_called()

    def test_auto_scene_position_retry_imports_latest_then_reruns_same_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retry_request = self._write_retry_workdir(root)
            write_manifest(
                root / "auto_task.json",
                {
                    "user_request": "生成一个未来汽车发布会展台",
                    "style_preset": "exhibition",
                    "quality_mode": "fast",
                    "geometry_mode": "strict",
                    "output_views": 3,
                    "num_candidates_per_view": 1,
                    "max_retries": 2,
                },
            )

            with patch("local3dai.cli.import_latest_codex_image2_results", return_value={"status": "complete", "import_count": 1}) as import_latest, patch(
                "local3dai.cli.run_auto_scene",
                return_value={
                    "status": "needs_review",
                    "white_model_position_lock": str(root / "reports" / "white_model_position_lock.json"),
                    "final_position_retry_plan": str(root / "reports" / "final_position_retry_plan.json"),
                },
            ) as run_scene:
                code, result = self._run_cli(
                    [
                        "auto-scene-run-position-retry",
                        "--workdir",
                        str(root),
                        "--backend",
                        "mock",
                        "--dry-run",
                        "--no-llm",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "needs_review")
            import_latest.assert_called_once()
            self.assertEqual(import_latest.call_args.args[0], retry_request)
            options = run_scene.call_args.args[0]
            self.assertEqual(options.output_dir, root)
            self.assertEqual(options.request, "生成一个未来汽车发布会展台")
            self.assertEqual(options.quality_mode, "fast")
            self.assertEqual(options.num_candidates_per_view, 1)
            self.assertTrue(options.dry_run)
            self.assertFalse(options.use_llm)

    def test_auto_scene_fit_position_lock_keeps_summary_final_images_on_originals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "renders" / "view_hero"
            view_dir.mkdir(parents=True)
            rgb = Image.new("RGB", (128, 128), (34, 36, 40))
            ImageDraw.Draw(rgb).rectangle((20, 48, 112, 112), fill=(232, 234, 238), outline=(80, 150, 230), width=3)
            rgb_path = view_dir / "rgb.png"
            rgb.save(rgb_path)
            mask = Image.new("L", (128, 128), 0)
            ImageDraw.Draw(mask).rectangle((20, 48, 112, 112), fill=255)
            mask_path = view_dir / "mask.png"
            mask.convert("RGB").save(mask_path)
            edge = Image.new("L", (128, 128), 0)
            ImageDraw.Draw(edge).rectangle((20, 48, 112, 112), outline=255, width=3)
            edge_path = view_dir / "edge.png"
            edge.convert("RGB").save(edge_path)
            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": [{"view_id": "view_hero", "files": {"rgb": str(rgb_path), "mask": str(mask_path), "edge": str(edge_path)}}]})

            final = root / "final" / "final_view_hero.png"
            final.parent.mkdir(parents=True)
            final_img = Image.new("RGB", (128, 128), (34, 36, 40))
            ImageDraw.Draw(final_img).rectangle((2, 4, 126, 78), fill=(235, 237, 240), outline=(80, 150, 230), width=3)
            final_img.save(final)
            write_manifest(
                root / "auto_scene_summary.json",
                {
                    "render_manifest": str(render_manifest),
                    "final_image": str(final),
                    "final_view_images": {"view_hero": str(final)},
                    "capabilities": {},
                },
            )

            code, result = self._run_cli(["auto-scene-fit-position-lock", "--workdir", str(root)])

            self.assertEqual(code, 0)
            self.assertEqual(result["mode"], "copy")
            self.assertNotEqual(result["output_view_images"]["view_hero"], str(final))
            summary = read_manifest(root / "auto_scene_summary.json")
            self.assertEqual(summary["final_image"], str(final))
            self.assertEqual(summary["final_view_images"], {"view_hero": str(final)})

    def test_auto_scene_position_retry_import_only_skips_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_retry_workdir(root)

            with patch("local3dai.cli.import_latest_codex_image2_results", return_value={"status": "complete", "import_count": 1}) as import_latest, patch(
                "local3dai.cli.run_auto_scene"
            ) as run_scene:
                code, result = self._run_cli(["auto-scene-run-position-retry", "--workdir", str(root), "--import-only"])

            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "imported")
            self.assertEqual(result["import"]["import_count"], 1)
            import_latest.assert_called_once()
            run_scene.assert_not_called()

    def test_auto_scene_plan_position_retry_backfills_existing_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "renders" / "view_hero"
            view_dir.mkdir(parents=True)
            files = {}
            for channel in ("rgb", "edge", "mask", "depth", "normal"):
                path = view_dir / f"{channel}.png"
                image = Image.new("RGB", (128, 128), (34, 36, 40))
                ImageDraw.Draw(image).rectangle((24, 42, 106, 92), fill=(232, 234, 238), outline=(80, 150, 230), width=3)
                image.save(path)
                files[channel] = str(path)
            final_image = root / "final" / "final_view_hero.png"
            final_image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (70, 70, 70)).save(final_image)
            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": [{"view_id": "view_hero", "files": files}]})
            write_manifest(
                root / "auto_scene_summary.json",
                {
                    "type": "auto_scene_summary",
                    "status": "needs_review",
                    "workdir": str(root),
                    "render_manifest": str(render_manifest),
                    "final_image": str(final_image),
                    "final_view_images": {"view_hero": str(final_image)},
                    "capabilities": {},
                },
            )

            code, result = self._run_cli(["auto-scene-plan-position-retry", "--workdir", str(root)])

            self.assertEqual(code, 0)
            self.assertEqual(result["retry_plan_status"], "awaiting_codex_image2_retry")
            self.assertTrue(Path(result["white_model_position_contract"]).exists())
            self.assertTrue(Path(result["white_model_position_lock"]).exists())
            self.assertTrue(Path(result["final_position_retry_plan"]).exists())
            final_request = read_manifest(root / "final" / "codex_image2_final_request.json")
            self.assertEqual(final_request["status"], "synthesized_for_position_retry")
            self.assertEqual(final_request["provider"], "codex_builtin_image2")
            retry_request = read_manifest(result["retry_request"])
            self.assertEqual(retry_request["status"], "awaiting_codex_image2")
            roles = {entry["role"] for entry in retry_request["requests"][0]["input_images"]}
            self.assertIn("white_model_rgb_position_lock", roles)
            self.assertIn("edge_silhouette_lock", roles)
            self.assertNotIn("negative_prompt", str(final_request))
            summary = read_manifest(root / "auto_scene_summary.json")
            self.assertEqual(summary["capabilities"]["final_position_retry_plan"]["status"], "awaiting_codex_image2_retry")

    def test_auto_scene_plan_position_retry_detects_missing_side_view_finals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = []
            for offset, view_id in enumerate(("view_hero", "view_left_30", "view_right_30")):
                view_dir = root / "renders" / view_id
                view_dir.mkdir(parents=True)
                files = {}
                for channel in ("rgb", "edge", "mask", "depth", "normal"):
                    path = view_dir / f"{channel}.png"
                    image = Image.new("RGB", (128, 128), (34, 36, 40))
                    ImageDraw.Draw(image).rectangle((24 + offset * 3, 42, 106 + offset * 3, 92), fill=(232, 234, 238), outline=(80, 150, 230), width=3)
                    image.save(path)
                    files[channel] = str(path)
                views.append({"view_id": view_id, "files": files})
            final_image = root / "final" / "final_view_hero.png"
            final_image.parent.mkdir(parents=True, exist_ok=True)
            hero = Image.new("RGB", (128, 128), (34, 36, 40))
            ImageDraw.Draw(hero).rectangle((24, 42, 106, 92), fill=(232, 234, 238), outline=(80, 150, 230), width=3)
            hero.save(final_image)
            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": views})
            write_manifest(
                root / "auto_scene_summary.json",
                {
                    "type": "auto_scene_summary",
                    "status": "needs_review",
                    "workdir": str(root),
                    "render_manifest": str(render_manifest),
                    "final_image": str(final_image),
                    "final_view_images": {"view_hero": str(final_image)},
                    "capabilities": {},
                },
            )

            code, result = self._run_cli(["auto-scene-plan-position-retry", "--workdir", str(root)])

            self.assertEqual(code, 0)
            self.assertEqual(result["white_model_position_lock_status"], "needs_review")
            self.assertEqual(result["retry_plan_status"], "awaiting_codex_image2_retry")
            lock_report = read_manifest(root / "reports" / "white_model_position_lock.json")
            self.assertEqual(lock_report["mode"], "multiview")
            self.assertIn("view_left_30", lock_report["failed_views"])
            self.assertIn("view_right_30", lock_report["failed_views"])
            retry_request = read_manifest(result["retry_request"])
            self.assertEqual(retry_request["request_count"], 3)
            self.assertEqual([item["view_id"] for item in retry_request["requests"]], ["view_hero", "view_left_30", "view_right_30"])
            self.assertNotIn("negative_prompt", json.dumps(retry_request, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
