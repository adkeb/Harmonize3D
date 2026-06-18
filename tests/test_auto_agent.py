from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from local3dai.auto_agent import AutoRunOptions, _call_qwen_planner, build_rule_plan, mesh_sanity_report, qwen_runtime_status, retry_policy_decisions, run_auto_agent, visual_judgement_report
from local3dai.webapp import _mark_auto_stage_artifacts, _start_auto_job, _stage_state
from PIL import Image


class AutoAgentTest(unittest.TestCase):

    def test_qwen_planner_sends_openai_tool_schema_and_parses_tool_calls(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "requirement_expander",
                                                "arguments": json.dumps({"reason": "parse request", "inputs": {"views": 3}}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        class FakeOpener:
            def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
                captured["timeout"] = timeout
                captured["payload"] = json.loads(request.data.decode("utf-8"))
                captured["authorization"] = request.get_header("Authorization")
                return FakeResponse()

        with patch.dict(os.environ, {"TEST_AGENT_KEY": "test-secret"}, clear=False), patch(
            "urllib.request.build_opener", return_value=FakeOpener()
        ):
            plan, info = _call_qwen_planner(
                {
                    "agent_llm": {
                        "enabled": True,
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "model": "qwen3.7-plus",
                        "timeout_seconds": 2,
                        "api_key_env": "TEST_AGENT_KEY",
                        "use_tool_schema": True,
                        "fallback_to_rules": False,
                    }
                },
                AutoRunOptions(request="生成一辆白色跑车", output_dir=Path("unused")),
            )
        payload = captured["payload"]
        self.assertEqual(info["backend"], "qwen_openai_compatible")
        self.assertEqual(info["tool_call_count"], 1)
        self.assertEqual(captured["authorization"], "Bearer test-secret")
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "auto")
        tool_names = {item["function"]["name"] for item in payload["tools"]}
        self.assertIn("visual_judgement", tool_names)
        self.assertIn("mesh_quality_check", tool_names)
        self.assertEqual(plan["tool_plan"][0]["tool"], "requirement_expander")

    def test_qwen_runtime_status_uses_no_proxy_model_probe(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            status = 200

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"data": [{"id": "qwen3.6-35b-a3b-nvfp4"}]}).encode("utf-8")

        class FakeOpener:
            def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
                captured["url"] = request.full_url
                captured["timeout"] = timeout
                return FakeResponse()

        with patch("urllib.request.build_opener", return_value=FakeOpener()) as build_opener:
            status = qwen_runtime_status(
                {
                    "agent_llm": {
                        "base_url": "http://127.0.0.1:8000/v1",
                        "model": "qwen3.6-35b-a3b-nvfp4",
                        "local_model_dir": "/root/sakura/models/Qwen3.6-35B-A3B-NVFP4",
                        "container_model_dir": "/root/.cache/huggingface/Qwen3.6-35B-A3B-NVFP4",
                        "no_proxy": True,
                    }
                },
                timeout=1.5,
            )
        self.assertTrue(status["ready"])
        self.assertEqual(status["service"]["models"], ["qwen3.6-35b-a3b-nvfp4"])
        self.assertIn("local_model", status)
        self.assertEqual(status["local_model"]["container_path"], "/root/.cache/huggingface/Qwen3.6-35B-A3B-NVFP4")
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/models")
        self.assertIsInstance(build_opener.call_args.args[0], urllib.request.ProxyHandler)

    def test_mesh_sanity_report_checks_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "white_mesh.obj"
            path.write_text("o sample\nv 0 0 0\n", encoding="utf-8")
            report = mesh_sanity_report(path, source_mode="procedural", expected_views=3)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["checks"]["exists"])
        self.assertTrue(report["checks"]["supported_extension"])

    def test_qwen_download_script_uses_hf_mirror_without_proxy(self) -> None:
        script = Path("scripts/download_qwen36_agent_model_direct.sh").read_text(encoding="utf-8")
        self.assertIn("nvidia/Qwen3.6-35B-A3B-NVFP4", script)
        self.assertIn("/root/sakura/models/Qwen3.6-35B-A3B-NVFP4", script)
        self.assertIn("HF_ENDPOINT=\"${HF_ENDPOINT:-https://hf-mirror.com}\"", script)
        self.assertIn("unset HTTP_PROXY HTTPS_PROXY ALL_PROXY", script)
        self.assertIn("NO_PROXY=\"*\"", script)
        self.assertIn("curl -I --noproxy '*'", script)
        self.assertIn("hf download", script)

    def test_qwen_service_script_matches_nvfp4_direct_runtime_contract(self) -> None:
        script = Path("scripts/start_qwen36_agent_service.sh").read_text(encoding="utf-8")
        self.assertIn("nvidia/Qwen3.6-35B-A3B-NVFP4", script)
        self.assertIn("LOCAL_MODEL_DIR", script)
        self.assertIn("CONTAINER_MODEL_DIR", script)
        self.assertIn("EFFECTIVE_MODEL_ID", script)
        self.assertIn("NVFP4_BACKEND", script)
        self.assertIn("--quantization", script)
        self.assertIn("modelopt", script)
        self.assertIn("HF_ENDPOINT=\"https://hf-mirror.com\"", script)
        self.assertIn("env -u HTTP_PROXY", script)
        self.assertIn("NO_PROXY=\"*\"", script)
        self.assertIn("--enable-auto-tool-choice", script)
        self.assertIn("--tool-call-parser", script)

    def test_rule_expander_and_camera_plan_emit_required_schema(self) -> None:
        options = AutoRunOptions(
            request="生成一辆未来感白色电动跑车，做三张产品图",
            output_dir=Path("unused"),
            output_views=3,
            source_mode="auto",
            use_llm=False,
        )
        task, prompt_plan, camera_plan = build_rule_plan({}, options)
        self.assertEqual(task["object_type"], "car")
        self.assertEqual(task["output_views"], 3)
        self.assertIn("render_prompt", prompt_plan)
        self.assertIn("no changed silhouette", prompt_plan["forbidden_changes"])
        self.assertEqual([view["view_id"] for view in camera_plan["views"]], ["view_locked", "view_left_30", "view_right_30"])
        self.assertEqual(camera_plan["camera_state"]["coordinate_space"], "blender_z_up")

    def test_retry_policy_maps_visual_score_failures_to_actions(self) -> None:
        decisions = retry_policy_decisions(
            {
                "silhouette_iou": 0.2,
                "added_part_penalty": 0.3,
                "roughness": 0.4,
                "background_cleanliness": 0.5,
            }
        )
        reasons = " ".join(item["reason"] for item in decisions)
        self.assertIn("silhouette_iou", reasons)
        self.assertIn("added_part_penalty", reasons)
        self.assertIn("roughness", reasons)
        self.assertIn("background_cleanliness", reasons)

    def test_visual_judgement_detects_nonblank_outputs_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "final.png"
            comparison = root / "comparison.png"
            contact = root / "contact.png"
            for path, color in [(final, (220, 30, 40)), (comparison, (30, 120, 220)), (contact, (60, 180, 80))]:
                image = Image.new("RGB", (64, 64), (245, 245, 245))
                for x in range(16, 48):
                    for y in range(16, 48):
                        image.putpixel((x, y), color)
                image.save(path)
            report = visual_judgement_report(
                final_image=str(final),
                comparison_image=str(comparison),
                contact_sheet=str(contact),
                agent_summary={
                    "structure_scores": {"view_locked": {"total": 0.8}},
                    "multiview_scores": {"total": 0.7},
                },
            )
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["has_visual_judgement"])
            self.assertTrue(report["checks"]["final_image_nonblank"])
            self.assertFalse(report["human_review_recommended"])
            self.assertTrue(report["final_image"]["nonblank"])

    def test_auto_run_dry_run_writes_tool_calls_and_visual_judgement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_auto_agent(
                AutoRunOptions(
                    request="生成一辆未来感白色电动跑车，做三张产品图",
                    output_dir=root,
                    config_path=Path("configs/local.json"),
                    source_mode="procedural",
                    backend="mock",
                    output_views=3,
                    num_candidates_per_view=1,
                    max_retries=0,
                    dry_run=True,
                    use_llm=False,
                )
            )
            self.assertIn(summary["status"], {"complete", "needs_review"})
            for key in ("auto_task", "prompt_plan", "camera_plan", "mesh_sanity", "agent_report", "visual_judgement", "tool_calls"):
                self.assertTrue(Path(summary[key]).exists(), key)
            visual = json.loads(Path(summary["visual_judgement"]).read_text(encoding="utf-8"))
            sanity = json.loads(Path(summary["mesh_sanity"]).read_text(encoding="utf-8"))
            tools = json.loads(Path(summary["tool_calls"]).read_text(encoding="utf-8"))
            self.assertTrue(visual["has_visual_judgement"])
            self.assertEqual(sanity["status"], "pass")
            tool_names = {item["tool"] for item in tools["calls"]}
            self.assertIn("generate_or_load_3d", tool_names)
            self.assertIn("mesh_quality_check", tool_names)
            self.assertIn("ai_candidate_search", tool_names)
            self.assertIn("visual_judgement", tool_names)
            self.assertTrue(summary["capabilities"]["tool_execution"]["enabled"])
            self.assertEqual(summary["capabilities"]["tool_execution"]["execution_mode"], "controlled_local_tools")
            self.assertIn("visual_judgement", summary["capabilities"]["tool_execution"]["executed_tool_names"])
            self.assertEqual(summary["capabilities"]["visual_judgement"]["status"], visual["status"])
            self.assertEqual(summary["capabilities"]["mesh_sanity"]["status"], "pass")

    def test_auto_stage_state_and_artifact_mapping_include_required_fields(self) -> None:
        stage = _stage_state("mesh_check")
        self.assertIn("retry_count", stage)
        job = {"stages": [_stage_state(stage_id) for stage_id in ("expand", "mesh_check", "complete")]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auto_task = root / "auto_task.json"
            sanity = root / "mesh" / "sanity.json"
            summary_path = root / "auto_summary.json"
            sanity.parent.mkdir(parents=True)
            auto_task.write_text("{}", encoding="utf-8")
            sanity.write_text("{}", encoding="utf-8")
            summary_path.write_text("{}", encoding="utf-8")
            _mark_auto_stage_artifacts(
                job,
                {
                    "workdir": str(root),
                    "auto_task": str(auto_task),
                    "mesh_sanity": str(sanity),
                    "visual_judgement": "",
                },
            )
        stages = {item["id"]: item for item in job["stages"]}
        self.assertIn("auto_task", stages["expand"]["artifacts"])
        self.assertIn("mesh_sanity", stages["mesh_check"]["artifacts"])
        self.assertIn("auto_summary", stages["complete"]["artifacts"])

    def test_web_auto_run_start_returns_task_snapshot(self) -> None:
        class DummyThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def start(self) -> None:
                pass

        with patch("local3dai.webapp.threading.Thread", DummyThread):
            result = _start_auto_job(
                {
                    "request": "生成一辆白色跑车",
                    "backend": "mock",
                    "dry_run": True,
                    "no_llm": True,
                    "output_views": 3,
                    "num_candidates_per_view": 1,
                }
            )
        self.assertTrue(result["task_id"].startswith("auto-"))
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["job"]["mode"], "auto_agent")
        self.assertIn("retry_count", result["job"]["stages"][0])


if __name__ == "__main__":
    unittest.main()
