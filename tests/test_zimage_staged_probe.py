from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_zimage_staged_control.py"
    spec = importlib.util.spec_from_file_location("probe_zimage_staged_control", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ZImageStagedProbeTest(unittest.TestCase):
    def _write_manifest(self, root: Path) -> Path:
        views = []
        for view_id in ("front_custom", "rear_custom"):
            view_dir = root / view_id
            view_dir.mkdir(parents=True)
            files: dict[str, str] = {}
            for channel, color in {
                "rgb": (220, 222, 224),
                "depth": (96, 96, 96),
                "edge": (255, 255, 255),
                "normal": (128, 128, 255),
                "mask": (255, 255, 255),
                "skeleton": (255, 255, 255),
            }.items():
                path = view_dir / f"{channel}.png"
                Image.new("RGB", (32, 32), color).save(path)
                files[channel] = str(path)
            views.append({"view_id": view_id, "camera": {}, "files": files})
        manifest = {"type": "render_manifest", "views": views}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_schedule_interpolation_and_wrapper_are_step_based(self) -> None:
        probe = _load_probe_module()
        schedule = probe.interpolate_schedule([(0.0, 1.0), (1.0, 0.0)], 5)
        self.assertEqual(schedule, [1.0, 0.75, 0.5, 0.25, 0.0])
        wrapper_result = probe._wrapper_self_test(schedule)
        self.assertTrue(wrapper_result["passed"])
        self.assertEqual(wrapper_result["observed"], schedule)

    def test_dry_run_uses_manifest_views_without_view_id_special_cases(self) -> None:
        probe = _load_probe_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_manifest(root)
            output = root / "probe"
            args = Namespace(
                manifest=str(manifest),
                output=str(output),
                base_model=str(root / "missing-base"),
                controlnet_file=str(root / "missing-controlnet.safetensors"),
                controlnet_config=str(root / "missing-controlnet-config"),
                i2l_model=str(root / "missing-i2l.safetensors"),
                appearance_lora="",
                prompt="red car",
                negative_prompt="lines",
                structure_channels=["depth", "gray", "canny"],
                structure_schedule="",
                appearance_schedule="",
                steps=8,
                width=64,
                height=64,
                seed=20260613,
                guidance_scale=1.0,
                device="cpu",
                dtype="bfloat16",
                max_views=1,
                mesh_lock_mode="none",
                detail_reference="",
                enable_model_cpu_offload=False,
                generate=False,
            )
            report = probe.run(args)
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["manifest"]["view_ids"], ["front_custom", "rear_custom"])
            self.assertEqual(len(report["structure_schedule"]), 8)
            self.assertEqual(len(report["appearance_schedule"]), 8)
            self.assertEqual(report["mesh_lock_mode"], "none")
            self.assertTrue(report["structure_schedule_wrapper_self_test"]["passed"])
            self.assertFalse(report["genericity_checks"]["view_id_specific_logic"])
            self.assertFalse(report["ready_to_generate"])
            for view in report["manifest"]["views"]:
                self.assertEqual(len(view["channels"]), 3)
                for channel in view["channels"]:
                    self.assertTrue(Path(channel["control_preview"]).exists())

    def test_path_status_marks_aria2_directories_incomplete(self) -> None:
        probe = _load_probe_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "model_index.json").write_text("{}", encoding="utf-8")
            marker = model / "weights.safetensors.aria2"
            marker.write_text("incomplete", encoding="utf-8")

            status = probe.path_status(model)

            self.assertTrue(status["exists"])
            self.assertFalse(status["complete"])
            self.assertEqual(status["incomplete_download_markers"], [str(marker)])


if __name__ == "__main__":
    unittest.main()
