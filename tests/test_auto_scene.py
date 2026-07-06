from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local3dai.auto_scene import (
    AutoSceneOptions,
    ExternalImagegenRequired,
    SceneToolExecutor,
    _binary_bbox,
    _extract_concept_camera_target,
    _mask_from_channel,
    _module_reference_prompt_with_safety,
    _pending_external_imagegen_summary,
    _score_camera_preview_image,
    apply_module_review_revisions,
    audit_auto_scene_image2_flow,
    assemble_scene,
    call_model_scene_planner,
    create_concept_final_comparison,
    create_final_position_retry_plan,
    create_white_model_multiview_position_lock_report,
    create_white_model_position_lock_report,
    create_white_model_position_contract,
    fit_final_images_to_white_model_positions,
    generate_concept_image,
    generate_codex_image2_final_render,
    generate_model_module_plan,
    generate_module_assets,
    generate_module_references,
    import_codex_image2_result,
    import_latest_codex_image2_results,
    module_failure_policy,
    module_presence_score,
    plan_scene_layout,
    repair_model_module_layout,
    render_scene_channels,
    review_module_reference_images,
    run_auto_scene,
    select_scene_camera,
    validate_module_layout_contract,
)
from local3dai.manifest import read_manifest, write_manifest
from PIL import Image, ImageDraw


class AutoSceneTest(unittest.TestCase):
    def _options(self, output_dir: Path | None = None) -> AutoSceneOptions:
        return AutoSceneOptions(
            request="生成一个未来汽车发布会展台，中央是一辆白色低趴电动超跑，周围有蓝色灯带、黑色展示台、两块发光屏幕、机械臂和灰色反光地面，输出三张产品级渲染图。",
            output_dir=output_dir or Path("unused"),
            config_path=Path("configs/local.json"),
            output_views=3,
            quality_mode="fast",
            geometry_mode="strict",
            style_preset="exhibition",
            backend="mock",
            num_candidates_per_view=1,
            max_retries=0,
            dry_run=True,
            use_llm=False,
        )

    def test_mock_scene_planner_builds_generic_scene_and_modules(self) -> None:
        plan, info = call_model_scene_planner({}, self._options())
        self.assertEqual(info["backend"], "mock_model_brain")
        self.assertEqual(plan["auto_task"]["mode"], "modular_scene_agent")
        self.assertEqual(plan["scene_plan"]["scene_type"], "model_planned_scene")
        hero_camera = plan["camera_plan"]["views"][0]
        self.assertEqual(hero_camera["camera_type"], "perspective")
        self.assertLessEqual(hero_camera["elevation_deg"], 14.0)
        modules = plan["module_plan"]["modules"]
        module_ids = {item["module_id"] for item in modules}
        self.assertGreaterEqual(len(modules), 3)
        self.assertIn("primary_subject", module_ids)
        self.assertIn("support_surface", module_ids)
        self.assertIn("background_element", module_ids)
        self.assertNotIn("main_vehicle", module_ids)

    def test_concept_plan_is_planning_only(self) -> None:
        plan, _ = call_model_scene_planner({}, self._options())
        concept = plan["concept_image_plan"]
        self.assertEqual(concept["output"], "concept/global_concept.png")
        self.assertIn("planning", concept["purpose"])
        self.assertNotIn("negative_prompt", concept)

    def test_module_plan_contains_prompts_roles_sizes_and_placement(self) -> None:
        plan, _ = call_model_scene_planner({}, self._options())
        module = next(item for item in plan["module_plan"]["modules"] if item["module_id"] == "primary_subject")
        self.assertEqual(module["role"], "hero_object")
        self.assertTrue(module["generate_reference_image"])
        self.assertTrue(module["generate_3d"])
        self.assertEqual(module["expected_real_world_size"]["unit"], "meters")
        self.assertEqual(module["placement"]["anchor"], "scene_center")
        self.assertIn("orthographic front view", module["reference_prompt"])

    def test_model_scene_planner_uses_dashscope_json_for_real_path(self) -> None:
        options = self._options()
        options.dry_run = False
        options.backend = None
        options.use_llm = True
        model_json = {
            "auto_task": {"expanded_request": "expanded by model", "output_views": 3},
            "scene_plan": {
                "scene_type": "model_showroom",
                "main_subject": {"name": "model planned car", "role": "hero_object", "priority": 1},
                "environment": {"description": "model planned stage"},
                "composition": {"camera_style": "low hero"},
                "global_style": {"color_palette": ["white", "blue"]},
            },
            "concept_image_plan": {
                "concept_prompt": "model generated concept prompt",
                "negative_prompt": "text, logos",
                "output": "concept/global_concept.png",
            },
            "prompt_plan": {"render_prompt": "model final prompt", "negative_prompt": "text"},
            "camera_plan": {"views": [{"view_id": "view_hero", "camera_type": "perspective", "elevation_deg": 8}]},
        }
        with patch("local3dai.auto_scene._call_dashscope_multimodal_json", return_value=(model_json, {"backend": "dashscope_multimodal"})) as call:
            plan, info = call_model_scene_planner({"agent_llm": {"enabled": True}}, options)
        self.assertEqual(info["plan_source"], "model_only")
        self.assertEqual(plan["auto_task"]["expanded_request"], "expanded by model")
        self.assertEqual(plan["scene_plan"]["scene_type"], "model_showroom")
        self.assertEqual(plan["concept_image_plan"]["concept_prompt"], "model generated concept prompt")
        self.assertNotIn("negative_prompt", plan["concept_image_plan"])
        self.assertNotIn("negative_prompt", plan["prompt_plan"])
        call.assert_called_once()

    def test_dashscope_multimodal_call_uses_timeout_and_disables_thinking(self) -> None:
        from types import SimpleNamespace

        import dashscope

        from local3dai.auto_scene import _call_dashscope_multimodal_json

        response = SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=[{"text": "{\"ok\": true}"}])
                    )
                ]
            ),
        )
        with patch("dashscope.MultiModalConversation.call", return_value=response) as call:
            parsed, info = _call_dashscope_multimodal_json(
                {"agent_llm": {"api_key": "test-key", "model": "qwen3.7-plus", "timeout_seconds": 7}},
                purpose="unit_test",
                messages=[{"role": "user", "content": [{"text": "return json"}]}],
            )
        self.assertTrue(parsed["ok"])
        self.assertEqual(info["backend"], "dashscope_multimodal")
        self.assertEqual(dashscope.base_http_api_url, "https://dashscope.aliyuncs.com/api/v1")
        self.assertEqual(call.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(call.call_args.kwargs["model"], "qwen3.7-plus")
        self.assertEqual(call.call_args.kwargs["messages"], [{"role": "user", "content": [{"text": "return json"}]}])
        self.assertEqual(call.call_args.kwargs["timeout"], 7)
        self.assertFalse(call.call_args.kwargs["enable_thinking"])

    def test_dashscope_imagegen_call_omits_negative_prompt(self) -> None:
        from types import SimpleNamespace

        from local3dai.auto_scene import _generate_dashscope_reference_image

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reference.png"
            concept = Path(tmp) / "concept.png"
            Image.new("RGB", (16, 16), (10, 20, 30)).save(concept)
            png = io.BytesIO()
            Image.new("RGB", (16, 16), (245, 245, 245)).save(png, format="PNG")
            response = SimpleNamespace(
                status_code=200,
                output={
                    "choices": [
                        {"message": {"content": [{"image": "https://example.test/reference.png"}]}}
                    ]
                },
            )
            with (
                patch("dashscope.aigc.image_generation.ImageGeneration.call", return_value=response) as call,
                patch("urllib.request.urlopen", return_value=io.BytesIO(png.getvalue())),
            ):
                result = _generate_dashscope_reference_image(
                    config={
                        "agent_llm": {"api_key": "test-key"},
                        "reference_generation": {"dashscope_model": "wan-test", "dashscope_image_edit_model": "wan-test-edit"},
                    },
                    prompt="front elevation single object on pure solid background",
                    output=output,
                    seed=1,
                    width=1024,
                    height=1024,
                    kind="module_reference",
                    module_id="unit_module",
                    source_image=concept,
                )
            self.assertTrue(output.exists())
            self.assertNotIn("negative_prompt", call.call_args.kwargs)
            self.assertEqual(call.call_args.kwargs["model"], "wan-test-edit")
            self.assertFalse(call.call_args.kwargs["enable_interleave"])
            message = call.call_args.kwargs["messages"][0]
            content = message.get("content") if hasattr(message, "get") else message["content"]
            self.assertIn("front elevation single object", content[0]["text"])
            self.assertEqual(content[1]["image"].split("://", 1)[0], "file")
            self.assertNotIn("negative_prompt", result)
            self.assertEqual(result["source_image"], str(concept.resolve()))
            metadata = read_manifest(output.with_suffix(".dashscope_imagegen.json"))
            self.assertNotIn("negative_prompt", metadata)
            self.assertEqual(metadata["source_image"], str(concept.resolve()))

    def test_model_scene_planner_missing_fields_uses_generic_not_hardcoded_fallback(self) -> None:
        options = self._options()
        options.dry_run = False
        options.backend = None
        options.use_llm = True
        model_json = {
            "auto_task": {"expanded_request": "minimal model output"},
            "concept_image_plan": {"concept_prompt": "minimal concept prompt"},
            "prompt_plan": {"render_prompt": "minimal final prompt"},
            "camera_plan": {"views": ["bad view", {"view_id": "view_hero", "azimuth_deg": 300}]},
        }
        with patch("local3dai.auto_scene._call_dashscope_multimodal_json", return_value=(model_json, {"backend": "dashscope_multimodal"})):
            plan, info = call_model_scene_planner({"agent_llm": {"enabled": True}}, options)
        self.assertEqual(info["plan_source"], "model_only")
        self.assertEqual(plan["scene_plan"]["scene_type"], "model_planned_scene")
        self.assertEqual(plan["module_plan"]["modules"], [])
        self.assertEqual(plan["camera_plan"]["views"], [{"view_id": "view_hero", "azimuth_deg": 300}])
        self.assertNotEqual(plan["scene_plan"]["scene_type"], "automotive_showroom")
        self.assertNotIn("main_vehicle", json.dumps(plan, ensure_ascii=False))

    def test_model_scene_planner_disabled_fails_real_run_without_rules(self) -> None:
        options = self._options()
        options.dry_run = False
        options.backend = None
        options.use_llm = True
        with self.assertRaisesRegex(RuntimeError, "scene_planner is disabled"):
            call_model_scene_planner({"agent_llm": {"enabled": False}}, options)

    def test_model_module_prompt_generation_normalizes_solid_background_prompts(self) -> None:
        options = self._options()
        options.dry_run = False
        options.backend = None
        options.use_llm = True
        with tempfile.TemporaryDirectory() as tmp:
            concept = Path(tmp) / "concept.png"
            Image.new("RGB", (16, 16), (80, 90, 100)).save(concept)
            model_json = {
                "modules": [
                    {
                        "module_id": "hero_car",
                        "name": "white concept car",
                        "role": "hero_object",
                        "reference_prompt": "white electric concept car",
                        "expected_real_world_size": {"width": 2.0, "depth": 4.8, "height": 1.2, "unit": "meters"},
                        "placement": {"anchor": "scene_center", "position": [0, 0, 0.4]},
                    }
                ]
            }
            with patch("local3dai.auto_scene._call_dashscope_multimodal_json", return_value=(model_json, {"backend": "dashscope_multimodal"})) as call:
                module_plan, info = generate_model_module_plan(
                    {"agent_llm": {"enabled": True}},
                    options,
                    scene_plan={"scene_type": "showroom"},
                    concept_image_plan={"concept_prompt": "concept"},
                    concept_image=concept,
                    concept_review={"status": "pass"},
                )
            self.assertEqual(info["backend"], "dashscope_multimodal")
            module = module_plan["modules"][0]
            self.assertEqual(module["role"], "hero_object")
            self.assertIn("solid", module["reference_prompt"])
            self.assertIn("blank unlabeled surfaces", module["reference_prompt"])
            self.assertNotIn("negative_prompt", module)
            self.assertIn("orthographic front view", module["reference_prompt"])
            sent_prompt = json.loads(call.call_args.kwargs["messages"][0]["content"][1]["text"])
            self.assertIn("Blender Z-up", sent_prompt["layout_coordinate_contract"]["coordinate_space"])
            self.assertGreaterEqual(len(sent_prompt["few_shot_layout_examples"]), 3)
            self.assertGreaterEqual(len(sent_prompt["few_shot_reference_prompt_examples"]), 3)
            self.assertIn("正交正视图", " ".join(sent_prompt["requirements"]))
            self.assertIn("三分之二视角", " ".join(sent_prompt["requirements"]))
            self.assertIn("不要 Markdown", " ".join(sent_prompt["requirements"]))
            self.assertIn("正向", " ".join(sent_prompt["requirements"]))
            self.assertNotIn("negative_prompt", sent_prompt["json_schema"]["modules"][0])
            self.assertGreaterEqual(call.call_args.kwargs["max_tokens"], 3600)

    def test_layout_repair_uses_model_instead_of_hardcoded_category_rules(self) -> None:
        options = self._options()
        options.dry_run = False
        options.backend = None
        options.use_llm = True
        module_plan = {
            "modules": [
                {
                    "module_id": "wide_wall_panel",
                    "name": "wide wall panel",
                    "category": "background_prop",
                    "role": "background",
                    "priority": 1,
                    "reference_prompt": "wide illuminated wall panel on solid background",
                    "expected_real_world_size": {"width": 8.0, "depth": 0.15, "height": 3.6, "unit": "meters"},
                    "placement": {"anchor": "background", "position": [0.0, 1.5, -6.0], "rotation_deg": [0.0, 0.0, 0.0]},
                    "generate_reference_image": True,
                    "generate_3d": True,
                }
            ]
        }
        initial_check = validate_module_layout_contract(module_plan)
        self.assertEqual(initial_check["status"], "needs_repair")
        with tempfile.TemporaryDirectory() as tmp:
            concept = Path(tmp) / "concept.png"
            Image.new("RGB", (16, 16), (80, 90, 100)).save(concept)
            repaired_json = {
                "modules": [
                    {
                        "module_id": "wide_wall_panel",
                        "placement": {"anchor": "background", "position": [0.0, 3.0, 1.8], "rotation_deg": [0.0, 0.0, 0.0]},
                    }
                ]
            }
            with patch("local3dai.auto_scene._call_dashscope_multimodal_json", return_value=(repaired_json, {"backend": "dashscope_multimodal"})) as call:
                repaired_plan, info = repair_model_module_layout(
                    {"agent_llm": {"enabled": True}},
                    options,
                    scene_plan={"scene_type": "showroom"},
                    concept_image=concept,
                    module_plan=module_plan,
                    layout_check=initial_check,
                )
            self.assertTrue(info["repair_attempted"])
            self.assertEqual(validate_module_layout_contract(repaired_plan)["status"], "pass")
            self.assertEqual(repaired_plan["modules"][0]["placement"]["position"], [0.0, 3.0, 1.8])
            sent_prompt = json.loads(call.call_args.kwargs["messages"][0]["content"][1]["text"])
            self.assertIn("few_shot_layout_examples", sent_prompt)
            self.assertEqual(sent_prompt["layout_issues_to_fix"]["status"], "needs_repair")

    def test_mechanical_arm_prompt_adds_hard_surface_safety_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = generate_module_references(
                root,
                {"global_style": {}},
                {
                    "modules": [
                        {
                            "module_id": "industrial_robotic_arm",
                            "name": "industrial robotic arm",
                            "reference_prompt": "sleek industrial robotic arm on white background",
                        }
                    ]
                },
                options=self._options(root),
            )
            manifest = read_manifest(root / "modules" / "industrial_robotic_arm" / "reference_manifest.json")
            self.assertEqual(len(summary["references"]), 1)
            self.assertIn("orthographic front view", manifest["prompt"])
            self.assertIn("simplified six-axis sealed articulated robot arm", manifest["prompt"])
            self.assertNotIn("six-axis industrial robotic arm", manifest["prompt"])
            self.assertNotIn("negative_prompt", manifest)
            duplicated_prompt = _module_reference_prompt_with_safety(
                {
                    "module_id": "industrial_robotic_arm",
                    "name": "industrial robotic arm",
                    "reference_prompt": "standard six-axis industrial robotic arm",
                },
                "standard six-axis industrial robotic arm, standard six-axis industrial robotic arm",
            )
            self.assertEqual(duplicated_prompt.count("simplified six-axis sealed articulated robot arm"), 1)
            self.assertNotIn("standard six-axis industrial robotic arm", duplicated_prompt)
            front_three_quarter = _module_reference_prompt_with_safety(
                {"module_id": "hero_car", "name": "hero car"},
                "sleek car, 3/4 front view, full body visible",
            )
            self.assertIn("orthographic front view", front_three_quarter)
            self.assertNotIn("3/4 front view", front_three_quarter)
            self.assertIn("front elevation", front_three_quarter)
            self.assertIn("zero yaw pitch roll camera", front_three_quarter)
            self.assertIn("flat frontal silhouette", front_three_quarter)
            self.assertNotIn("no diagonal pose", front_three_quarter)
            self.assertNotIn("no floor plane", front_three_quarter)
            self.assertNotIn("no contact shadow", front_three_quarter)
            side_profile = _module_reference_prompt_with_safety(
                {"module_id": "industrial_robotic_arm", "name": "industrial robotic arm"},
                "metal industrial robotic arm, side profile view, full object visible",
            )
            self.assertIn("orthographic front view", side_profile)
            self.assertIn("front-facing catalog cutout", side_profile)
            self.assertNotIn("not side profile", side_profile)
            self.assertNotIn("not three-quarter view", side_profile)
            self.assertNotIn("no loose cables", side_profile)
            self.assertNotIn("no external wires", side_profile)
            self.assertNotIn("no visible cable harnesses", side_profile)
            self.assertIn("untextured matte clay 3D CAD reconstruction input", side_profile)
            self.assertIn("untextured matte white/gray 3D CAD clay model", side_profile)
            self.assertIn("simplified six-axis sealed articulated robot arm", side_profile)
            self.assertIn("simple geometric primitive shape", side_profile)
            self.assertIn("toy-like educational robot arm", side_profile)
            self.assertIn("five smooth cylinders connected by circular hinge disks only", side_profile)
            self.assertIn("made only from smooth cylinders", side_profile)
            self.assertIn("sealed-joint design", side_profile)
            self.assertIn("smooth uninterrupted exterior", side_profile)
            self.assertNotIn("black corrugated", side_profile)
            self.assertNotIn("electronic boxes", side_profile)
            self.assertNotIn("exposed connectors", side_profile)
            self.assertNotIn("industrial robotic arm", side_profile.lower())
            self.assertNotIn("side profile view", side_profile)
            self.assertNotIn(", not,", side_profile)
            cable_conflict = _module_reference_prompt_with_safety(
                {"module_id": "industrial_robotic_arm", "name": "industrial robotic arm"},
                "metal robot arm, cables neatly routed along the arm structure, reflection on floor",
            )
            self.assertNotIn("cables neatly routed", cable_conflict)
            self.assertNotIn("reflection on floor", cable_conflict)
            self.assertNotIn("cable", cable_conflict.lower())
            self.assertIn("smooth uninterrupted exterior", cable_conflict)
            side_revision = _module_reference_prompt_with_safety(
                {"module_id": "industrial_robotic_arm", "name": "industrial robotic arm"},
                "strict orthographic SIDE PROFILE view, 90-degree side elevation, flat side silhouette",
            )
            self.assertIn("front elevation", side_revision)
            self.assertIn("front-facing catalog cutout", side_revision)
            self.assertNotIn("side profile", side_revision.lower())
            self.assertNotIn("side elevation", side_revision.lower())
            revised_with_negative_views = _module_reference_prompt_with_safety(
                {"module_id": "screen", "name": "display screen"},
                "display screen, not angled view, not three-quarter view, not side profile, not top view, not rear view, not back view",
            )
            self.assertNotIn(", not,", revised_with_negative_views)
            self.assertNotRegex(revised_with_negative_views, r"(?:^|, )not(?:,|$)")
            self.assertNotIn("ABB", side_profile)
            self.assertNotIn("KUKA", side_profile)

            platform_prompt = _module_reference_prompt_with_safety(
                {"module_id": "black_display_platform", "name": "matte black display platform", "role": "supporting_object"},
                "rectangular matte black display platform base",
            )
            self.assertIn("smooth hard-surface neutral gray CAD model", platform_prompt)
            self.assertIn("one-piece low rectangular cuboid slab", platform_prompt)
            self.assertIn("single-layer undivided horizontal slab", platform_prompt)
            self.assertIn("height about one tenth of total width", platform_prompt)
            self.assertIn("crisp straight 90-degree edges", platform_prompt)
            self.assertIn("long low rectangular silhouette", platform_prompt)
            self.assertNotIn("matte black", platform_prompt.lower())
            self.assertNotIn("display platform", platform_prompt.lower())
    def test_module_review_revision_rejects_mechanical_arm_water_prompt(self) -> None:
        module_plan = {
            "modules": [
                {
                    "module_id": "industrial_robotic_arm",
                    "name": "industrial robotic arm",
                    "reference_prompt": "sleek metallic industrial robotic arm",
                }
            ]
        }
        failed = module_plan["modules"][0]["module_id"]
        apply_module_review_revisions(
            module_plan,
            {
                "failed_modules": [failed],
                "module_reviews": [
                    {
                        "module_id": failed,
                        "status": "revise",
                        "revised_reference_prompt": "A transparent liquid water robotic arm sculpture",
                    }
                ],
            },
        )
        prompt = module_plan["modules"][0]["reference_prompt"].lower()
        self.assertIn("simple geometric sealed articulated robot arm cad primitive", prompt)
        self.assertIn("five smooth cylinders connected by circular hinge disks only", prompt)
        self.assertNotIn("liquid water", prompt)

    def test_module_reference_review_prompt_forbids_side_profile_revision(self) -> None:
        options = self._options()
        options.dry_run = False
        options.backend = None
        options.use_llm = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept.png"
            reference = root / "reference.png"
            Image.new("RGB", (32, 32), (220, 220, 220)).save(concept)
            Image.new("RGB", (32, 32), (240, 240, 240)).save(reference)
            module_plan = {
                "modules": [
                    {
                        "module_id": "robot_arm",
                        "name": "robot arm",
                        "role": "supporting_object",
                        "reference_prompt": "front elevation robot arm",
                    }
                ]
            }
            reference_summary = {"references": [{"module_id": "robot_arm", "reference_image": str(reference)}]}
            with patch(
                "local3dai.auto_scene._call_dashscope_multimodal_json",
                return_value=(
                    {
                        "status": "pass",
                        "failed_modules": [],
                        "module_reviews": [{"module_id": "robot_arm", "status": "pass"}],
                    },
                    {"backend": "dashscope_multimodal"},
                ),
            ) as call:
                review = review_module_reference_images(
                    {"agent_llm": {"enabled": True}},
                    options,
                    module_plan=module_plan,
                    reference_summary=reference_summary,
                    concept_image=concept,
                    attempt=0,
                    output_path=root / "review.json",
                )
            self.assertEqual(review["status"], "pass")
            sent = call.call_args.kwargs["messages"][0]["content"][-1]["text"]
            prompt = json.loads(sent)
            joined = " ".join(prompt["pass_criteria"])
            self.assertIn("front elevation", joined)
            self.assertIn("轻微圆柱厚度", joined)
            self.assertIn("只含正向目标描述", joined)

    def test_scene_layout_outputs_explainable_transforms(self) -> None:
        plan, _ = call_model_scene_planner({}, self._options())
        assets = {
            "modules": [
                {
                    "module_id": module["module_id"],
                    "model_path": f"modules/{module['module_id']}/model.glb",
                    "bbox": module["expected_real_world_size"],
                }
                for module in plan["module_plan"]["modules"]
            ]
        }
        assembly = plan_scene_layout(plan["scene_plan"], plan["module_plan"], assets)
        self.assertEqual(assembly["coordinate_system"], "blender_z_up")
        self.assertEqual(assembly["units"], "meters")
        self.assertFalse(assembly["collision_report"]["has_major_collision"])
        first = assembly["modules"][0]
        self.assertIn("transform", first)
        self.assertIn("layout_reason", first)
        self.assertEqual(len(first["transform"]["scale"]), 3)

    def test_module_presence_score_schema(self) -> None:
        assembly = {
            "modules": [
                {"module_id": "main_vehicle", "visibility_priority": 1},
                {"module_id": "left_led_screen", "visibility_priority": 3},
            ]
        }
        score = module_presence_score(assembly, {"views": []}, {"structure_scores": {}})
        self.assertGreater(score["total"], 0.0)
        self.assertEqual(score["module_scores"][0]["module_id"], "main_vehicle")
        self.assertIn("position_adherence", score["module_scores"][0])

    def test_module_failure_policy_hero_fails_and_background_fallbacks(self) -> None:
        hero = {"module_id": "main_vehicle", "role": "hero_object"}
        platform = {"module_id": "display_platform", "role": "supporting_object"}
        self.assertEqual(module_failure_policy(hero, "missing mesh", allow_fallback=True)["action"], "fail_task")
        self.assertEqual(module_failure_policy(platform, "missing mesh", allow_fallback=True)["action"], "use_procedural_proxy")

    def test_image2_concept_and_module_references_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept" / "global_concept.png"
            concept.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), (201, 12, 34)).save(concept)
            generated_concept = generate_concept_image(root, {"output": "concept/global_concept.png", "seed": 7})
            self.assertEqual(generated_concept, concept)
            self.assertEqual(Image.open(concept).convert("RGB").getpixel((0, 0)), (201, 12, 34))

            reference = root / "modules" / "main_vehicle" / "reference.png"
            reference.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), (21, 178, 66)).save(reference)
            summary = generate_module_references(
                root,
                {"global_style": {"color_palette": ["white", "blue"]}},
                {"modules": [{"module_id": "main_vehicle", "reference_prompt": "single car reference"}]},
            )
            manifest = read_manifest(root / "modules" / "main_vehicle" / "reference_manifest.json")
            self.assertEqual(summary["references"][0]["image_source"], "image2_provided")
            self.assertEqual(manifest["created_by"], "image2_provided")
            self.assertEqual(Image.open(reference).convert("RGB").getpixel((0, 0)), (21, 178, 66))
            self.assertTrue((root / "modules" / "main_vehicle" / "preprocessed.png").exists())

    def test_real_concept_generation_requires_external_imagegen_when_image_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = self._options(root)
            options.dry_run = False
            options.backend = None
            with self.assertRaisesRegex(RuntimeError, "External image2/imagegen generation required"):
                generate_concept_image(
                    root,
                    {"output": "concept/global_concept.png", "concept_prompt": "agent concept prompt", "negative_prompt": "text"},
                    config={"reference_generation": {"provider": "external_imagegen"}},
                    options=options,
                )
            request = read_manifest(root / "concept" / "imagegen_request.json")
            self.assertEqual(request["kind"], "concept")
            self.assertEqual(request["prompt"], "agent concept prompt")
            self.assertEqual(request["status"], "awaiting_external_imagegen")
            self.assertEqual(request["provider"], "codex_builtin_image2")
            self.assertTrue(Path(request["request_path"]).is_absolute())
            self.assertIn("auto-scene-import-image2", request["import_command"])
            self.assertIn("auto-scene-import-latest-image2", request["latest_import_command"])
            self.assertNotIn("negative_prompt", request)
            self.assertTrue(Path(request["output_path"]).is_absolute())
            handoff = Path(request["codex_image2_handoff"])
            self.assertTrue(handoff.is_absolute())
            self.assertTrue(handoff.exists())
            handoff_text = handoff.read_text(encoding="utf-8")
            self.assertIn("agent concept prompt", handoff_text)
            self.assertIn(request["output_path"], handoff_text)
            self.assertIn(request["request_path"], handoff_text)
            self.assertIn("auto-scene-import-image2", handoff_text)
            self.assertIn("auto-scene-import-latest-image2", handoff_text)
            self.assertNotIn("negative_prompt", handoff_text)

    def test_concept_generation_can_use_dashscope_imagegen_file_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = self._options(root)
            options.dry_run = False
            options.backend = None

            def fake_dashscope_image(**kwargs: object) -> dict[str, object]:
                output = Path(kwargs["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (24, 24), (240, 240, 255)).save(output)
                return {
                    "created_by": "dashscope_imagegen_reference_generation",
                    "image_source": "dashscope_imagegen",
                    "path": str(output),
                    "metadata": str(output.with_suffix(".dashscope_imagegen.json")),
                }

            with patch("local3dai.auto_scene._generate_dashscope_reference_image", side_effect=fake_dashscope_image) as call:
                output = generate_concept_image(
                    root,
                    {"output": "concept/global_concept.png", "concept_prompt": "agent concept prompt", "negative_prompt": "text"},
                    config={"reference_generation": {"provider": "dashscope_imagegen"}},
                    options=options,
                )
            manifest = read_manifest(output.with_name("generation_manifest.json"))
            self.assertTrue(output.exists())
            self.assertEqual(manifest["created_by"], "dashscope_imagegen_reference_generation")
            self.assertTrue(Path(manifest["path"]).is_absolute())
            self.assertTrue(Path(manifest["prompt_file"]).is_absolute())
            self.assertEqual(call.call_args.kwargs["kind"], "concept")
            self.assertNotIn("negative_prompt", call.call_args.kwargs)
            self.assertNotIn("source_image", call.call_args.kwargs)

    def test_auto_scene_returns_pending_summary_for_external_concept_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            write_manifest(config_path, {"reference_generation": {"provider": "external_imagegen"}, "models": {"hunyuan3d_2_1_shape": {"enabled": False}}})
            options = self._options(root)
            options.config_path = config_path
            options.dry_run = False
            options.backend = None
            options.use_llm = True
            plan = {
                "auto_task": {
                    "task_id": "model-planned-pending-concept",
                    "user_request": options.request,
                    "expanded_request": "model expanded scene for pending concept image",
                    "style_preset": options.style_preset,
                    "quality_mode": options.quality_mode,
                    "geometry_mode": options.geometry_mode,
                    "output_views": 1,
                    "num_candidates_per_view": 1,
                    "max_retries": 0,
                },
                "scene_plan": {
                    "scene_type": "model_planned_scene",
                    "main_subject": {"name": "model planned hero object", "role": "hero_object", "priority": 1},
                    "environment": {"description": "model planned environment"},
                    "composition": {"camera_style": "model planned camera"},
                    "global_style": {"material_language": "clean CGI", "color_palette": ["white", "blue"]},
                    "expected_elements": ["model planned hero object"],
                },
                "concept_image_plan": {
                    "concept_prompt": "model-written concept prompt awaiting codex image2",
                    "width": 1024,
                    "height": 1024,
                    "output": "concept/global_concept.png",
                },
                "prompt_plan": {"render_prompt": "model-written final render prompt"},
                "camera_plan": {"views": [{"view_id": "view_hero", "camera_type": "perspective", "elevation_deg": 8}]},
            }
            with patch("local3dai.auto_scene.call_model_scene_planner", return_value=(plan, {"backend": "dashscope_multimodal", "plan_source": "model_only"})):
                summary = run_auto_scene(options)
            self.assertEqual(summary["status"], "awaiting_external_imagegen")
            self.assertEqual(summary["stage"], "concept_image_generation")
            self.assertTrue(Path(summary["external_imagegen"]["request_path"]).exists())
            self.assertEqual(read_manifest(root / "auto_scene_summary.json")["status"], "awaiting_external_imagegen")
            stages = read_manifest(summary["stages"])
            by_id = {stage["id"]: stage for stage in stages["stages"]}
            self.assertEqual(stages["status"], "awaiting_external_imagegen")
            self.assertEqual(by_id["concept"]["status"], "awaiting_external_imagegen")
            self.assertTrue(by_id["concept"]["message"])
            self.assertIn("concept_image_request", by_id["concept"]["artifacts"])
            self.assertIn("error", by_id["concept"])
            self.assertIn("retry_count", by_id["concept"])

    def test_auto_scene_resume_reuses_existing_scene_and_module_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            write_manifest(config_path, {"reference_generation": {"provider": "external_imagegen"}, "models": {"hunyuan3d_2_1_shape": {"enabled": False}}})
            options = self._options(root)
            options.config_path = config_path
            options.dry_run = False
            options.backend = None
            options.use_llm = True
            plan = {
                "auto_task": {
                    "task_id": "model-planned-resume-scene",
                    "user_request": options.request,
                    "expanded_request": "model expanded scene for resume",
                    "style_preset": options.style_preset,
                    "quality_mode": options.quality_mode,
                    "geometry_mode": options.geometry_mode,
                    "output_views": 1,
                    "num_candidates_per_view": 1,
                    "max_retries": 0,
                },
                "scene_plan": {
                    "scene_type": "model_planned_scene",
                    "main_subject": {"name": "model planned hero object", "role": "hero_object", "priority": 1},
                    "environment": {"description": "model planned environment"},
                    "composition": {"camera_style": "model planned camera"},
                    "global_style": {"material_language": "clean CGI", "color_palette": ["white", "blue"]},
                    "expected_elements": ["model planned hero object"],
                },
                "concept_image_plan": {
                    "concept_prompt": "model-written concept prompt awaiting codex image2",
                    "width": 1024,
                    "height": 1024,
                    "output": "concept/global_concept.png",
                },
                "prompt_plan": {"render_prompt": "model-written final render prompt"},
                "camera_plan": {"views": [{"view_id": "view_hero", "camera_type": "perspective", "elevation_deg": 8}]},
            }
            module_plan = {
                "modules": [
                    {
                        "module_id": "model_hero_object",
                        "name": "model planned hero object",
                        "category": "vehicle",
                        "role": "hero_object",
                        "priority": 1,
                        "reference_prompt": "single centered object, strict orthographic front view, pure solid background",
                        "expected_real_world_size": {"width": 2.0, "depth": 3.0, "height": 1.2, "unit": "meters"},
                        "placement": {"anchor": "scene_center", "position": [0.0, 0.0, 0.6], "rotation_deg": [0.0, 0.0, 0.0], "scale_policy": "model_planned"},
                        "generate_reference_image": True,
                        "generate_3d": True,
                    }
                ]
            }

            with patch("local3dai.auto_scene.call_model_scene_planner", return_value=(plan, {"backend": "dashscope_multimodal", "plan_source": "model_only"})):
                first = run_auto_scene(options)
            self.assertEqual(first["stage"], "concept_image_generation")
            self.assertTrue((root / "prompt_plan.json").exists())
            self.assertTrue((root / "cameras" / "camera_plan.json").exists())
            concept_output = Path(first["external_imagegen"]["output_path"])
            concept_output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (220, 230, 245)).save(concept_output)

            with (
                patch("local3dai.auto_scene.call_model_scene_planner", side_effect=AssertionError("scene planner must not be recalled")),
                patch(
                    "local3dai.auto_scene.review_concept_image",
                    return_value={"type": "concept_image_review", "status": "pass", "checks": {"required_elements_present": True}},
                ),
                patch("local3dai.auto_scene.generate_model_module_plan", return_value=(module_plan, {"backend": "dashscope_multimodal"})) as module_prompt,
            ):
                second = run_auto_scene(options)
            self.assertEqual(second["stage"], "module_reference_generation")
            self.assertEqual(module_prompt.call_count, 1)
            batch = read_manifest(root / "modules" / "imagegen_batch_request.json")
            self.assertEqual(batch["kind"], "module_reference_batch")
            self.assertEqual(batch["requests"][0]["source_image"], str(concept_output))
            self.assertEqual(batch["provider"], "codex_builtin_image2")
            self.assertTrue(Path(batch["request_path"]).is_absolute())
            self.assertIn("auto-scene-import-image2", batch["import_command"])
            self.assertIn("auto-scene-import-latest-image2", batch["latest_import_command"])
            handoff = Path(batch["codex_image2_handoff"])
            self.assertTrue(handoff.exists())
            handoff_text = handoff.read_text(encoding="utf-8")
            self.assertIn("strict orthographic front view", handoff_text)
            self.assertIn(str(concept_output), handoff_text)
            self.assertIn(batch["requests"][0]["output_path"], handoff_text)
            self.assertIn("model_hero_object=/path/to/codex-image2-output-1.png", handoff_text)
            self.assertIn("auto-scene-import-latest-image2", handoff_text)
            self.assertNotIn("negative_prompt", json.dumps(batch, ensure_ascii=False))

            with (
                patch("local3dai.auto_scene.call_model_scene_planner", side_effect=AssertionError("scene planner must not be recalled")),
                patch(
                    "local3dai.auto_scene.review_concept_image",
                    return_value={"type": "concept_image_review", "status": "pass", "checks": {"required_elements_present": True}},
                ),
                patch("local3dai.auto_scene.generate_model_module_plan", side_effect=AssertionError("module prompt planner must not be recalled")),
            ):
                third = run_auto_scene(options)
            self.assertEqual(third["stage"], "module_reference_generation")
            self.assertEqual(third["planning"]["plan_source"], "existing_scene_planner_outputs")
            calls = [item["tool"] for item in read_manifest(root / "reports" / "tool_calls.json")["calls"]]
            self.assertIn("module_prompt_generation", calls)

    def test_auto_scene_dashscope_imagegen_provider_runs_without_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            write_manifest(
                config_path,
                {
                    "ai": {"default_backend": "mock"},
                    "auto_scene": {"final_render_provider": "local_agent"},
                    "reference_generation": {"provider": "dashscope_imagegen", "concurrent_workers": 4, "max_review_attempts": 3},
                    "models": {"hunyuan3d_2_1_shape": {"enabled": False}, "flux2_klein_4b": {"backend": "mock", "enabled": True}},
                    "agent_llm": {"enabled": True},
                },
            )
            options = self._options(root)
            options.config_path = config_path
            options.dry_run = False
            options.backend = None
            options.use_llm = True
            options.render_backend = "procedural"
            options.allow_procedural_fallback = True
            plan = {
                "auto_task": {
                    "task_id": "model-planned-test-scene",
                    "user_request": options.request,
                    "expanded_request": "model-expanded compact product stage",
                    "style_preset": options.style_preset,
                    "quality_mode": options.quality_mode,
                    "geometry_mode": options.geometry_mode,
                    "output_views": 1,
                    "num_candidates_per_view": 1,
                    "max_retries": 0,
                },
                "scene_plan": {
                    "scene_type": "model_planned_scene",
                    "main_subject": {"name": "model planned hero object", "role": "hero_object", "priority": 1},
                    "environment": {"description": "model planned clean exhibition stage"},
                    "composition": {"camera_style": "model selected hero view"},
                    "global_style": {"material_language": "clean CGI", "color_palette": ["white", "graphite", "blue"], "avoid": ["text"]},
                    "expected_elements": ["model planned hero object"],
                },
                "concept_image_plan": {
                    "concept_prompt": "model-written concept prompt for a compact clean exhibition stage with one hero object",
                    "negative_prompt": "text, logo, people",
                    "width": 1024,
                    "height": 1024,
                    "output": "concept/global_concept.png",
                },
                "module_plan": {
                    "modules": [
                        {
                            "module_id": "model_hero_object",
                            "name": "model planned hero object",
                            "category": "product",
                            "role": "hero_object",
                            "priority": 1,
                            "reference_prompt": "single model planned hero object",
                            "negative_prompt": "text, logo, people, environment",
                            "expected_real_world_size": {"width": 2.0, "depth": 3.0, "height": 1.2, "unit": "meters"},
                            "placement": {"anchor": "scene_center", "position": [0.0, 0.0, 0.6], "rotation_deg": [0.0, 0.0, 0.0], "scale_policy": "model_planned"},
                            "generate_reference_image": True,
                            "generate_3d": True,
                        }
                    ]
                },
                "prompt_plan": {"render_prompt": "model-written final render prompt", "negative_prompt": "text, logo, people"},
                "camera_plan": {
                    "views": [
                        {
                            "view_id": "view_hero",
                            "role": "model selected hero",
                            "azimuth_deg": 320.0,
                            "elevation_deg": 8.0,
                            "camera_type": "perspective",
                            "focal_length_mm": 58.0,
                            "distance_scale": 1.0,
                            "target": [0.0, 0.0, 0.0],
                        }
                    ]
                },
            }

            def fake_dashscope_image(**kwargs: object) -> dict[str, object]:
                output = Path(kwargs["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (230, 238, 248)).save(output)
                return {
                    "created_by": "dashscope_imagegen_reference_generation",
                    "image_source": "dashscope_imagegen",
                    "path": str(output),
                    "metadata": str(output.with_suffix(".dashscope_imagegen.json")),
                }

            review_attempts: list[int] = []

            def fake_module_review(*args: object, **kwargs: object) -> dict[str, object]:
                attempt = int(kwargs["attempt"])
                review_attempts.append(attempt)
                if attempt < 2:
                    review = {
                        "type": "module_reference_review",
                        "attempt": attempt,
                        "status": "revise",
                        "failed_modules": ["model_hero_object"],
                        "module_reviews": [
                            {
                                "module_id": "model_hero_object",
                                "status": "revise",
                                "revised_reference_prompt": f"retry {attempt} strict orthographic front view hero object",
                                "notes": "test requested regeneration",
                            }
                        ],
                    }
                    write_manifest(Path(kwargs["output_path"]), review)
                    return review
                review = {
                    "type": "module_reference_review",
                    "attempt": attempt,
                    "status": "pass",
                    "failed_modules": [],
                    "module_reviews": [
                        {"module_id": item["module_id"], "status": "pass"}
                        for item in kwargs["reference_summary"].get("references", [])
                    ],
                }
                write_manifest(Path(kwargs["output_path"]), review)
                return review

            with (
                patch("local3dai.auto_scene.call_model_scene_planner", return_value=(plan, {"backend": "dashscope_multimodal", "plan_source": "model_only"})) as planner,
                patch("local3dai.auto_scene.generate_model_module_plan", return_value=(plan["module_plan"], {"backend": "dashscope_multimodal"})),
                patch(
                    "local3dai.auto_scene.review_concept_image",
                    return_value={
                        "type": "concept_image_review",
                        "status": "pass",
                        "checks": {"required_elements_present": True, "no_forbidden_text": True, "composition_usable": True},
                        "missing_elements": [],
                        "extra_elements": [],
                    },
                ),
                patch(
                    "local3dai.auto_scene.review_module_reference_images",
                    side_effect=fake_module_review,
                ),
                patch("local3dai.auto_scene.repair_model_module_layout", return_value=(plan["module_plan"], {"backend": "test_noop"})),
                patch("local3dai.auto_scene._generate_dashscope_reference_image", side_effect=fake_dashscope_image) as imagegen,
            ):
                summary = run_auto_scene(options)
            self.assertNotEqual(summary["status"], "awaiting_external_imagegen")
            self.assertTrue(planner.called)
            self.assertEqual(review_attempts, [0, 1, 2])
            self.assertEqual(imagegen.call_count, 4)
            concept_generation = read_manifest(Path(summary["global_concept"]).with_name("generation_manifest.json"))
            self.assertEqual(concept_generation["created_by"], "dashscope_imagegen_reference_generation")
            self.assertTrue(Path(concept_generation["path"]).is_absolute())
            self.assertTrue(Path(summary["global_concept"]).is_absolute())
            module_manifest = read_manifest(root / "modules" / "model_hero_object" / "reference_manifest.json")
            self.assertEqual(module_manifest["image_source"], "dashscope_imagegen")
            self.assertNotIn("negative_prompt", module_manifest)
            self.assertEqual(Path(module_manifest["source_image"]), Path(summary["global_concept"]))
            self.assertEqual(module_manifest["review_status"], "pass")
            self.assertEqual(module_manifest["review_attempt"], 2)
            self.assertTrue(Path(module_manifest["reference_image"]).is_absolute())
            self.assertTrue(Path(module_manifest["prompt_file"]).is_absolute())
            self.assertTrue(Path(summary["final_image"]).is_absolute())
            self.assertTrue(Path(summary["final_image"]).exists())
            self.assertTrue(Path(summary["image2_flow_audit"]).exists())
            image2_audit = read_manifest(summary["image2_flow_audit"])
            self.assertEqual(image2_audit["type"], "auto_scene_image2_flow_audit")
            self.assertEqual(summary["capabilities"]["image2_flow_audit"]["status"], image2_audit["status"])
            tool_calls = [item["tool"] for item in read_manifest(summary["tool_calls"])["calls"]]
            self.assertIn("image2_flow_audit", tool_calls)

    def test_real_module_reference_requires_external_imagegen_when_image_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = self._options(root)
            options.dry_run = False
            options.backend = None
            with self.assertRaisesRegex(RuntimeError, "External image2/imagegen generation required"):
                generate_module_references(
                    root,
                    {"global_style": {}},
                    {"modules": [{"module_id": "main_vehicle", "reference_prompt": "agent object prompt"}]},
                    config={"reference_generation": {"provider": "external_imagegen"}},
                    options=options,
                )
            request = read_manifest(root / "modules" / "main_vehicle" / "imagegen_request.json")
            self.assertEqual(request["kind"], "module_reference")
            self.assertEqual(request["module_id"], "main_vehicle")
            self.assertEqual(request["provider"], "codex_builtin_image2")
            self.assertIn("auto-scene-import-image2", request["import_command"])
            self.assertIn("auto-scene-import-latest-image2", request["latest_import_command"])
            self.assertNotIn("negative_prompt", request)
            self.assertTrue(Path(request["codex_image2_handoff"]).exists())
            batch = read_manifest(root / "modules" / "imagegen_batch_request.json")
            self.assertEqual(batch["kind"], "module_reference_batch")
            self.assertEqual(batch["request_count"], 1)
            self.assertIn("main_vehicle=/path/to/codex-image2-output-1.png", batch["import_command"])
            self.assertIn("auto-scene-import-latest-image2", batch["latest_import_command"])
            self.assertTrue(Path(batch["codex_image2_handoff"]).exists())

    def test_import_codex_image2_single_request_copies_and_marks_fulfilled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "generated" / "concept.png"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (20, 80, 180)).save(source)
            output = root / "concept" / "global_concept.png"
            request_path = write_manifest(
                root / "concept" / "imagegen_request.json",
                {
                    "type": "external_imagegen_request",
                    "status": "awaiting_external_imagegen",
                    "provider": "codex_builtin_image2",
                    "kind": "concept",
                    "output_path": str(output),
                    "prompt": "global concept prompt",
                },
            )

            summary = import_codex_image2_result(request_path, image_path=source)

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["import_count"], 1)
            self.assertTrue(output.exists())
            with Image.open(output) as imported:
                self.assertEqual(imported.size, (32, 32))
            request = read_manifest(request_path)
            self.assertEqual(request["status"], "fulfilled_by_codex_image2")
            self.assertEqual(request["fulfilled_output_path"], str(output.resolve()))
            self.assertTrue(Path(request["import_manifest"]).exists())
            self.assertNotIn("negative_prompt", json.dumps(request, ensure_ascii=False))

    def test_import_codex_image2_batch_maps_module_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            car_source = root / "generated" / "car.png"
            screen_source = root / "generated" / "screen.png"
            car_source.parent.mkdir(parents=True)
            Image.new("RGB", (24, 24), (240, 240, 245)).save(car_source)
            Image.new("RGB", (24, 24), (30, 70, 120)).save(screen_source)
            car_output = root / "modules" / "main_vehicle" / "reference.png"
            screen_output = root / "modules" / "display_screen" / "reference.png"
            batch_path = write_manifest(
                root / "modules" / "imagegen_batch_request.json",
                {
                    "type": "external_imagegen_batch_request",
                    "status": "awaiting_external_imagegen",
                    "provider": "codex_builtin_image2",
                    "kind": "module_reference_batch",
                    "requests": [
                        {"kind": "module_reference", "module_id": "main_vehicle", "output_path": str(car_output)},
                        {"kind": "module_reference", "module_id": "display_screen", "output_path": str(screen_output)},
                    ],
                },
            )

            summary = import_codex_image2_result(
                batch_path,
                image_mappings={"main_vehicle": car_source, "display_screen": screen_source},
            )

            self.assertEqual(summary["import_count"], 2)
            self.assertTrue(car_output.exists())
            self.assertTrue(screen_output.exists())
            batch = read_manifest(batch_path)
            self.assertEqual(batch["status"], "fulfilled_by_codex_image2")
            self.assertTrue(all(item["status"] == "fulfilled_by_codex_image2" for item in batch["requests"]))
            self.assertEqual(batch["requests"][0]["fulfilled_output_path"], str(car_output.resolve()))
            self.assertEqual(batch["requests"][1]["fulfilled_output_path"], str(screen_output.resolve()))
            self.assertNotIn("negative_prompt", json.dumps(batch, ensure_ascii=False))

    def test_import_latest_codex_image2_single_request_scans_generated_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / "codex_home"
            generated = codex_home / "generated_images" / "thread"
            generated.mkdir(parents=True)
            marker = root / "before.marker"
            marker.write_text("before\n", encoding="utf-8")
            os.utime(marker, (1000, 1000))
            generated_image = generated / "ig_concept.png"
            Image.new("RGB", (32, 20), (80, 120, 210)).save(generated_image)
            os.utime(generated_image, (1005, 1005))
            output = root / "concept" / "global_concept.png"
            request_path = write_manifest(
                root / "concept" / "imagegen_request.json",
                {
                    "type": "external_imagegen_request",
                    "status": "awaiting_external_imagegen",
                    "provider": "codex_builtin_image2",
                    "kind": "concept",
                    "output_path": str(output),
                    "prompt": "concept prompt",
                },
            )

            summary = import_latest_codex_image2_results(request_path, codex_home=codex_home, after_marker=marker)

            self.assertEqual(summary["source"], "codex_generated_images_latest_scan")
            self.assertEqual(summary["codex_generated_images"], [str(generated_image.resolve())])
            self.assertTrue(output.exists())
            request = read_manifest(request_path)
            self.assertEqual(request["status"], "fulfilled_by_codex_image2")
            self.assertEqual(request["imported_image"], str(generated_image.resolve()))
            self.assertNotIn("negative_prompt", json.dumps(request, ensure_ascii=False))

    def test_import_latest_codex_image2_batch_maps_request_order_to_mtime_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / "codex_home"
            generated = codex_home / "generated_images" / "thread"
            generated.mkdir(parents=True)
            first_image = generated / "ig_first.png"
            second_image = generated / "ig_second.png"
            Image.new("RGB", (18, 18), (220, 220, 230)).save(first_image)
            Image.new("RGB", (18, 18), (30, 70, 130)).save(second_image)
            os.utime(first_image, (2001, 2001))
            os.utime(second_image, (2002, 2002))
            car_output = root / "modules" / "main_vehicle" / "reference.png"
            screen_output = root / "modules" / "display_screen" / "reference.png"
            batch_path = write_manifest(
                root / "modules" / "imagegen_batch_request.json",
                {
                    "type": "external_imagegen_batch_request",
                    "status": "awaiting_external_imagegen",
                    "provider": "codex_builtin_image2",
                    "kind": "module_reference_batch",
                    "requests": [
                        {"kind": "module_reference", "module_id": "main_vehicle", "output_path": str(car_output)},
                        {"kind": "module_reference", "module_id": "display_screen", "output_path": str(screen_output)},
                    ],
                },
            )

            summary = import_latest_codex_image2_results(batch_path, codex_home=codex_home)

            self.assertEqual(summary["codex_generated_images"], [str(first_image.resolve()), str(second_image.resolve())])
            batch = read_manifest(batch_path)
            self.assertEqual(batch["requests"][0]["imported_image"], str(first_image.resolve()))
            self.assertEqual(batch["requests"][1]["imported_image"], str(second_image.resolve()))
            self.assertTrue(car_output.exists())
            self.assertTrue(screen_output.exists())
            self.assertEqual(batch["status"], "fulfilled_by_codex_image2")
            self.assertNotIn("negative_prompt", json.dumps(batch, ensure_ascii=False))

    def test_audit_auto_scene_image2_flow_passes_strict_codex_hunyuan_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept" / "global_concept.png"
            concept.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (90, 120, 200)).save(concept)
            write_manifest(
                root / "concept" / "concept_image_plan.json",
                {"concept_prompt": "model-expanded concept prompt", "output": "concept/global_concept.png"},
            )
            write_manifest(
                concept.with_name("generation_manifest.json"),
                {"type": "concept_image_generation", "created_by": "imagegen_skill_external", "image_source": "imagegen_skill_external", "path": str(concept)},
            )
            write_manifest(
                root / "concept" / "imagegen_request.json",
                {"provider": "codex_builtin_image2", "status": "fulfilled_by_codex_image2", "output_path": str(concept)},
            )
            write_manifest(
                root / "concept" / "concept_review.json",
                {"type": "concept_image_review", "backend": "dashscope_multimodal", "status": "pass"},
            )
            module_id = "main_vehicle"
            reference = root / "modules" / module_id / "reference.png"
            reference.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (230, 232, 238)).save(reference)
            model_path = root / "modules" / module_id / "model.glb"
            model_path.write_bytes(b"glTF test")
            metadata_path = root / "modules" / module_id / "metadata.json"
            write_manifest(
                root / "modules" / "module_plan.json",
                {
                    "modules": [
                        {
                            "module_id": module_id,
                            "role": "hero_object",
                            "reference_prompt": "single object, strict orthographic front view, pure solid background",
                            "generate_3d": True,
                        }
                    ]
                },
            )
            write_manifest(root / "modules" / "module_prompt_info.json", {"backend": "dashscope_multimodal", "model": "qwen3.7-plus"})
            write_manifest(
                reference.with_name("reference_manifest.json"),
                {
                    "module_id": module_id,
                    "reference_image": str(reference),
                    "image_source": "imagegen_skill_external",
                    "created_by": "imagegen_skill_external",
                    "review_status": "pass",
                    "source_image": str(concept),
                },
            )
            write_manifest(
                root / "modules" / "module_reference_review.json",
                {"type": "module_reference_review", "backend": "dashscope_multimodal", "status": "pass", "failed_modules": []},
            )
            write_manifest(
                metadata_path,
                {"module_id": module_id, "model_path": str(model_path), "created_by": "hunyuan3d_2_1_shape_from_reviewed_reference", "source_reference": str(reference)},
            )
            write_manifest(
                root / "modules" / "module_asset_manifest.json",
                {
                    "modules": [
                        {
                            "module_id": module_id,
                            "model_path": str(model_path),
                            "metadata": str(metadata_path),
                            "fallback_used": False,
                            "sanity": {"proxy_geometry": "hunyuan3d_2_1_shape_from_reviewed_reference", "fallback_used": False},
                        }
                    ]
                },
            )
            final_render_dir = root / "renders" / "view_hero"
            final_render_dir.mkdir(parents=True)
            final_input_images = []
            for role, channel in (
                ("white_model_rgb_position_lock", "rgb"),
                ("edge_silhouette_lock", "edge"),
                ("mask_composition_reference", "mask"),
            ):
                channel_path = final_render_dir / f"{channel}.png"
                Image.new("RGB", (32, 32), (218, 220, 224)).save(channel_path)
                final_input_images.append({"role": role, "channel": channel, "path": str(channel_path)})
            position_contract = {
                "view_id": "view_hero",
                "status": "pass",
                "source_rgb": str(final_render_dir / "rgb.png"),
                "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                "center_norm": [0.5, 0.5],
                "coverage_ratio": 1.0,
            }
            write_manifest(
                root / "reports" / "white_model_position_contract.json",
                {"type": "white_model_position_contract", "status": "pass", "contracts": [position_contract]},
            )
            write_manifest(
                root / "final" / "codex_image2_final_request.json",
                {
                    "type": "codex_image2_final_render_request",
                    "provider": "codex_builtin_image2",
                    "white_model_position_contract": str(root / "reports" / "white_model_position_contract.json"),
                    "planning_images_excluded_from_final_inputs": [str(concept.resolve())],
                    "requests": [
                        {
                            "view_id": "view_hero",
                            "input_images": final_input_images,
                            "position_lock": {"primary_reference_role": "white_model_rgb_position_lock"},
                            "position_lock_contract": position_contract,
                        }
                    ],
                },
            )
            write_manifest(
                root / "auto_scene_summary.json",
                {
                    "type": "auto_scene_summary",
                    "status": "needs_review",
                    "planning": {"backend": "dashscope_multimodal", "model": "qwen3.7-plus", "plan_source": "model_only", "rule_planning": "disabled_for_real_runs"},
                    "concept_image_plan": str(root / "concept" / "concept_image_plan.json"),
                    "global_concept": str(concept),
                    "concept_review": str(root / "concept" / "concept_review.json"),
                    "module_plan": str(root / "modules" / "module_plan.json"),
                    "module_prompt_info": str(root / "modules" / "module_prompt_info.json"),
                    "module_reference_review": str(root / "modules" / "module_reference_review.json"),
                    "module_asset_manifest": str(root / "modules" / "module_asset_manifest.json"),
                },
            )

            report = audit_auto_scene_image2_flow(root)

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["passed_required_checks"], report["required_checks"])
            passed = {check["id"] for check in report["checks"] if check["status"] == "pass"}
            self.assertIn("final_image2_uses_white_model_channels", passed)
            self.assertIn("final_image2_excludes_planning_images", passed)
            self.assertIn("final_image2_has_white_model_position_contract", passed)
            self.assertTrue((root / "reports" / "image2_flow_audit.json").exists())

    def test_audit_auto_scene_image2_flow_fails_non_codex_reference_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept" / "global_concept.png"
            concept.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), (90, 120, 200)).save(concept)
            write_manifest(root / "concept" / "concept_image_plan.json", {"concept_prompt": "prompt"})
            write_manifest(concept.with_name("generation_manifest.json"), {"created_by": "dashscope_imagegen_reference_generation", "image_source": "dashscope_imagegen"})
            write_manifest(root / "concept" / "concept_review.json", {"backend": "dashscope_multimodal", "status": "pass"})
            write_manifest(root / "modules" / "module_plan.json", {"modules": []})
            write_manifest(
                root / "auto_scene_summary.json",
                {
                    "planning": {"backend": "dashscope_multimodal", "model": "qwen3.7-plus", "plan_source": "model_only", "rule_planning": "disabled_for_real_runs"},
                    "concept_image_plan": str(root / "concept" / "concept_image_plan.json"),
                    "global_concept": str(concept),
                    "concept_review": str(root / "concept" / "concept_review.json"),
                    "module_plan": str(root / "modules" / "module_plan.json"),
                },
            )

            report = audit_auto_scene_image2_flow(root)
            failed = {check["id"] for check in report["checks"] if check["status"] == "fail"}

            self.assertEqual(report["status"], "fail")
            self.assertIn("concept_image_codex_image2", failed)

    def test_import_codex_image2_final_request_maps_view_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_source = root / "generated" / "view_hero.png"
            final_source.parent.mkdir(parents=True)
            Image.new("RGB", (40, 28), (180, 190, 210)).save(final_source)
            output = root / "final" / "view_hero.png"
            request_path = write_manifest(
                root / "final" / "codex_image2_final_request.json",
                {
                    "type": "codex_image2_final_render_request",
                    "status": "awaiting_codex_image2",
                    "provider": "codex_builtin_image2",
                    "requests": [
                        {
                            "view_id": "view_hero",
                            "output_path": str(output),
                            "prompt": "render final image using white model position lock",
                        }
                    ],
                },
            )

            summary = import_codex_image2_result(request_path, image_mappings={"view_hero": final_source})

            self.assertEqual(summary["request_kind"], "codex_image2_final_render_request")
            self.assertTrue(output.exists())
            request = read_manifest(request_path)
            self.assertEqual(request["status"], "fulfilled_by_codex_image2")
            self.assertEqual(request["requests"][0]["fulfilled_output_path"], str(output.resolve()))
            self.assertNotIn("negative_prompt", json.dumps(request, ensure_ascii=False))

    def test_module_references_can_use_dashscope_imagegen_file_provider(self) -> None:
        import threading
        import time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = self._options(root)
            options.dry_run = False
            options.backend = None
            modules = [
                {"module_id": "main_vehicle", "reference_prompt": "single car reference"},
                {"module_id": "display_platform", "reference_prompt": "single platform reference"},
            ]
            lock = threading.Lock()
            active_calls = 0
            max_active_calls = 0

            def fake_dashscope_image(**kwargs: object) -> dict[str, object]:
                nonlocal active_calls, max_active_calls
                with lock:
                    active_calls += 1
                    max_active_calls = max(max_active_calls, active_calls)
                time.sleep(0.05)
                with lock:
                    active_calls -= 1
                output = Path(kwargs["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (24, 24), (230, 240, 250)).save(output)
                return {
                    "created_by": "dashscope_imagegen_reference_generation",
                    "image_source": "dashscope_imagegen",
                    "path": str(output),
                    "metadata": str(output.with_suffix(".dashscope_imagegen.json")),
                }

            with patch("local3dai.auto_scene._generate_dashscope_reference_image", side_effect=fake_dashscope_image) as call:
                summary = generate_module_references(
                    root,
                    {"global_style": {}},
                    {"modules": modules},
                    config={"reference_generation": {"provider": "dashscope_imagegen", "concurrent_workers": 4}},
                    options=options,
                )
            self.assertEqual(summary["concurrent_workers"], 2)
            self.assertEqual(len(summary["references"]), 2)
            self.assertEqual(call.call_count, 2)
            for image_call in call.call_args_list:
                self.assertNotIn("negative_prompt", image_call.kwargs)
                self.assertEqual(image_call.kwargs["source_image"], None)
            self.assertGreaterEqual(max_active_calls, 2)
            manifest = read_manifest(root / "modules" / "main_vehicle" / "reference_manifest.json")
            self.assertEqual(manifest["created_by"], "dashscope_imagegen_reference_generation")
            self.assertEqual(manifest["image_source"], "dashscope_imagegen")
            self.assertNotIn("negative_prompt", manifest)
            self.assertTrue(Path(manifest["reference_image"]).is_absolute())
            self.assertTrue(Path(manifest["preprocessed_image"]).is_absolute())
            self.assertTrue(Path(manifest["prompt_file"]).is_absolute())

    def test_pending_batch_summary_exposes_absolute_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_path = root / "modules" / "imagegen_batch_request.json"
            handoff_path = root / "modules" / "codex_image2_batch_handoff.md"
            handoff_path.parent.mkdir(parents=True)
            handoff_path.write_text("# handoff\n", encoding="utf-8")
            write_manifest(
                batch_path,
                {
                    "kind": "module_reference_batch",
                    "status": "awaiting_external_imagegen",
                    "codex_image2_handoff": str(handoff_path),
                    "requests": [
                        {"output_path": str(root / "modules" / "a" / "reference.png")},
                        {"output_path": str(root / "modules" / "b" / "reference.png")},
                    ],
                },
            )
            summary = _pending_external_imagegen_summary(
                workdir=root,
                started=0.0,
                options=self._options(root),
                tool_executor=SceneToolExecutor(),
                request_error=ExternalImagegenRequired(batch_path),
                stage="module_reference_generation",
            )
            output_paths = summary["external_imagegen"]["output_paths"]
            self.assertEqual(len(output_paths), 2)
            self.assertTrue(all(Path(path).is_absolute() for path in output_paths))
            self.assertEqual(summary["external_imagegen"]["codex_image2_handoff"], str(handoff_path.resolve()))

    def test_codex_image2_final_render_writes_white_model_position_lock_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept" / "global_concept.png"
            concept.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (40, 80, 160)).save(concept)
            view_dir = root / "renders" / "view_hero"
            view_dir.mkdir(parents=True)
            files = {}
            for channel in ("rgb", "edge", "depth", "normal", "mask"):
                path = view_dir / f"{channel}.png"
                image = Image.new("RGB", (32, 32), (42, 44, 48))
                ImageDraw.Draw(image).rectangle((7, 10, 26, 24), fill=(210, 210, 210), outline=(120, 160, 220), width=1)
                image.save(path)
                files[channel] = str(path)
            render_manifest = write_manifest(
                root / "renders" / "render_manifest.json",
                {"views": [{"view_id": "view_hero", "files": files}]},
            )
            with self.assertRaises(ExternalImagegenRequired) as raised:
                generate_codex_image2_final_render(
                    workdir=root,
                    render_manifest_path=render_manifest,
                    prompt_plan={"render_prompt": "premium final showroom render"},
                    concept_image=concept,
                    output_views=1,
                )
            request = read_manifest(raised.exception.request_path)
            self.assertEqual(request["type"], "codex_image2_final_render_request")
            self.assertEqual(request["status"], "awaiting_codex_image2")
            self.assertEqual(request["provider"], "codex_builtin_image2")
            self.assertEqual(request["request_count"], 1)
            self.assertTrue(Path(request["request_path"]).is_absolute())
            self.assertTrue(Path(request["codex_image2_handoff"]).exists())
            self.assertIn("auto-scene-import-image2", request["import_command"])
            self.assertIn("auto-scene-import-latest-image2", request["latest_import_command"])
            self.assertNotIn("negative_prompt", json.dumps(request, ensure_ascii=False))
            item = request["requests"][0]
            self.assertTrue(Path(item["output_path"]).is_absolute())
            self.assertTrue(Path(request["white_model_position_contract"]).exists())
            self.assertEqual(item["position_lock_contract"]["status"], "pass")
            self.assertEqual(len(item["position_lock_contract"]["bbox_norm"]), 4)
            self.assertEqual(len(item["position_lock_contract"]["center_norm"]), 2)
            roles = {entry["role"] for entry in item["input_images"]}
            self.assertIn("white_model_rgb_position_lock", roles)
            self.assertIn("edge_silhouette_lock", roles)
            self.assertIn("mask_composition_reference", roles)
            self.assertNotIn("appearance_style_reference_only", roles)
            self.assertNotIn(str(concept.resolve()), json.dumps(item["input_images"], ensure_ascii=False))
            self.assertEqual(request["planning_images_excluded_from_final_inputs"], [str(concept.resolve())])
            self.assertEqual(item["position_lock"]["style_source"], "prompt_plan.render_prompt")
            self.assertEqual(item["position_lock"]["contract_source"], request["white_model_position_contract"])
            self.assertIn("exact geometry", item["prompt"])
            self.assertIn("White-model position contract", item["prompt"])
            prompt_lower = item["prompt"].lower()
            self.assertNotIn("hypercar", prompt_lower)
            self.assertNotIn("hero car", prompt_lower)
            self.assertNotIn("robot arm", prompt_lower)
            self.assertNotIn("automotive launch booth", prompt_lower)
            self.assertEqual(item["position_lock"]["primary_reference_role"], "white_model_rgb_position_lock")
            handoff_text = Path(request["codex_image2_handoff"]).read_text(encoding="utf-8")
            self.assertIn("white_model_rgb_position_lock", handoff_text)
            self.assertIn("White-model position contract", handoff_text)
            self.assertIn("view_hero=/path/to/codex-image2-output-1.png", handoff_text)
            self.assertIn("auto-scene-import-latest-image2", handoff_text)

    def test_blender_render_backend_writes_auto_scene_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_model = root / "scene" / "final_scene.glb"
            scene_model.parent.mkdir(parents=True)
            scene_model.write_bytes(b"glTF")
            scene_assembly = write_manifest(root / "scene" / "scene_assembly.json", {"modules": [{"module_id": "main_vehicle"}]})
            camera_plan = {
                "views": [
                    {"view_id": "view_hero", "azimuth_deg": 310.0, "elevation_deg": 8.0, "camera_type": "perspective", "focal_length_mm": 58.0},
                    {"view_id": "view_left_30", "azimuth_deg": 280.0, "elevation_deg": 8.0, "camera_type": "perspective", "focal_length_mm": 58.0},
                    {"view_id": "view_right_30", "azimuth_deg": 340.0, "elevation_deg": 8.0, "camera_type": "perspective", "focal_length_mm": 58.0},
                ]
            }

            def fake_render_model_with_blender(**kwargs: object) -> Path:
                output = Path(kwargs["output_dir"])
                views = []
                for view_id in ("view_locked", "view_left_30", "view_right_30"):
                    view_dir = output / view_id
                    view_dir.mkdir(parents=True, exist_ok=True)
                    files = {}
                    for channel in ("rgb", "depth", "edge", "normal", "mask"):
                        path = view_dir / f"{channel}.png"
                        Image.new("RGB", (8, 8), (80, 90, 100)).save(path)
                        files[channel] = str(path)
                    views.append({"view_id": view_id, "files": files})
                return write_manifest(output / "manifest.json", {"type": "render_manifest", "source": str(kwargs["model_path"]), "views": views})

            with patch("local3dai.auto_scene.render_model_with_blender", side_effect=fake_render_model_with_blender):
                manifest_path = render_scene_channels(
                    root,
                    {"scene_model_path": str(scene_model)},
                    scene_assembly,
                    camera_plan,
                    views=3,
                    config={"paths": {"blender_script": "blender_scripts/batch_render.py"}, "render": {}, "system": {}},
                    dry_run=True,
                    render_backend="blender",
                    allow_fallback=False,
                )
            manifest = read_manifest(manifest_path)
            self.assertEqual(manifest["source"], "auto_scene_blender_render_channels")
            self.assertEqual(manifest["render_backend"], "blender")
            self.assertEqual(manifest["views"][0]["view_id"], "view_hero")
            self.assertEqual(manifest["views"][0]["original_render_view_id"], "view_locked")
            self.assertEqual(manifest["views"][0]["module_ids_visible"], ["main_vehicle"])

    def test_concept_final_comparison_flags_bad_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept.png"
            final = root / "final.png"
            render_rgb = root / "render" / "view_hero" / "rgb.png"
            render_rgb.parent.mkdir(parents=True)
            concept_img = Image.new("RGB", (256, 256), (38, 42, 48))
            concept_img.paste((235, 238, 240), (50, 116, 206, 176))
            concept_img.paste((20, 132, 245), (20, 30, 52, 210))
            concept_img.save(concept)
            Image.new("RGB", (256, 256), (188, 188, 188)).save(final)
            Image.new("RGB", (256, 256), (220, 220, 220)).save(render_rgb)
            render_manifest = write_manifest(
                root / "render" / "render_manifest.json",
                {"views": [{"view_id": "view_hero", "files": {"rgb": str(render_rgb)}}]},
            )
            report = create_concept_final_comparison(
                concept_image=concept,
                final_image=final,
                render_manifest=render_manifest,
                output_image=root / "final" / "concept_vs_final.png",
                output_report=root / "reports" / "concept_final_comparison.json",
            )
            self.assertEqual(report["status"], "needs_review")
            self.assertIn("visual_entropy", report["failure_reasons"])
            self.assertTrue((root / "final" / "concept_vs_final.png").exists())

    def test_white_model_position_lock_report_flags_layout_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "renders" / "view_hero"
            view_dir.mkdir(parents=True)
            rect = (82, 148, 430, 312)
            rgb = Image.new("RGB", (512, 512), (224, 228, 232))
            ImageDraw.Draw(rgb).rectangle(rect, fill=(238, 240, 242), outline=(30, 36, 44), width=4)
            rgb_path = view_dir / "rgb.png"
            rgb.save(rgb_path)
            mask = Image.new("L", (512, 512), 0)
            ImageDraw.Draw(mask).rectangle(rect, fill=255)
            mask_path = view_dir / "mask.png"
            mask.convert("RGB").save(mask_path)
            edge = Image.new("L", (512, 512), 0)
            ImageDraw.Draw(edge).rectangle(rect, outline=255, width=5)
            edge_path = view_dir / "edge.png"
            edge.convert("RGB").save(edge_path)
            render_manifest = write_manifest(
                root / "renders" / "render_manifest.json",
                {"views": [{"view_id": "view_hero", "files": {"rgb": str(rgb_path), "mask": str(mask_path), "edge": str(edge_path)}}]},
            )
            aligned = root / "final" / "aligned.png"
            aligned.parent.mkdir(parents=True)
            aligned_img = Image.new("RGB", (512, 512), (224, 228, 232))
            ImageDraw.Draw(aligned_img).rectangle(rect, fill=(245, 246, 247), outline=(32, 40, 50), width=4)
            aligned_img.save(aligned)
            aligned_report = create_white_model_position_lock_report(
                final_image=aligned,
                render_manifest=render_manifest,
                output_report=root / "reports" / "aligned_lock.json",
                output_image=root / "final" / "aligned_lock.png",
            )
            self.assertEqual(aligned_report["status"], "pass")
            self.assertGreaterEqual(aligned_report["metrics"]["total"], 0.58)
            self.assertTrue((root / "final" / "aligned_lock.png").exists())

            shifted = root / "final" / "shifted.png"
            shifted_img = Image.new("RGB", (512, 512), (224, 228, 232))
            ImageDraw.Draw(shifted_img).rectangle((238, 56, 500, 182), fill=(245, 246, 247), outline=(32, 40, 50), width=4)
            shifted_img.save(shifted)
            shifted_report = create_white_model_position_lock_report(
                final_image=shifted,
                render_manifest=render_manifest,
                output_report=root / "reports" / "shifted_lock.json",
                output_image=root / "final" / "shifted_lock.png",
            )
            self.assertEqual(shifted_report["status"], "needs_review")
            self.assertIn("center_alignment", shifted_report["failure_reasons"])

    def test_soft_gray_mask_channel_does_not_select_entire_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_path = Path(tmp) / "soft_mask.png"
            image = Image.new("L", (128, 128), 61)
            ImageDraw.Draw(image).rectangle((30, 42, 100, 90), fill=214)
            image.convert("RGB").save(mask_path)

            mask = _mask_from_channel(mask_path, size=(128, 128))
            bbox = _binary_bbox(mask)

            self.assertIsNotNone(bbox)
            self.assertLess(float(mask.mean()), 0.5)
            self.assertEqual(bbox, [0.234375, 0.328125, 0.789062, 0.710938])

    def test_fit_final_image_to_white_model_position_restores_bbox_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "renders" / "view_left_30"
            view_dir.mkdir(parents=True)
            rect = (20, 48, 112, 112)
            rgb = Image.new("RGB", (128, 128), (34, 36, 40))
            ImageDraw.Draw(rgb).rectangle(rect, fill=(232, 234, 238), outline=(80, 150, 230), width=3)
            rgb_path = view_dir / "rgb.png"
            rgb.save(rgb_path)
            mask = Image.new("L", (128, 128), 0)
            ImageDraw.Draw(mask).rectangle(rect, fill=255)
            mask_path = view_dir / "mask.png"
            mask.convert("RGB").save(mask_path)
            edge = Image.new("L", (128, 128), 0)
            ImageDraw.Draw(edge).rectangle(rect, outline=255, width=3)
            edge_path = view_dir / "edge.png"
            edge.convert("RGB").save(edge_path)
            render_manifest = write_manifest(
                root / "renders" / "render_manifest.json",
                {"views": [{"view_id": "view_left_30", "files": {"rgb": str(rgb_path), "mask": str(mask_path), "edge": str(edge_path)}}]},
            )

            final = root / "final" / "drifted.png"
            final.parent.mkdir(parents=True)
            final_img = Image.new("RGB", (128, 128), (34, 36, 40))
            ImageDraw.Draw(final_img).rectangle((2, 4, 126, 78), fill=(235, 237, 240), outline=(80, 150, 230), width=3)
            final_img.save(final)

            fit = fit_final_images_to_white_model_positions(
                final_view_images={"view_left_30": final},
                render_manifest=render_manifest,
                output_report=root / "reports" / "white_model_position_fit.json",
                output_dir=root / "final" / "position_fitted",
            )
            self.assertEqual(fit["status"], "pass")
            fitted = Path(fit["output_view_images"]["view_left_30"])
            self.assertTrue(fitted.exists())

            lock = create_white_model_multiview_position_lock_report(
                final_view_images={"view_left_30": fitted},
                render_manifest=render_manifest,
                output_report=root / "reports" / "white_model_position_lock.json",
                output_image=root / "final" / "white_position_lock_overlay.png",
            )
            self.assertEqual(lock["status"], "pass")
            self.assertGreaterEqual(lock["view_reports"][0]["metrics"]["center_alignment"], 0.9)

    def test_multiview_position_lock_fails_when_side_view_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = []
            final_view_images = {}
            for index, view_id in enumerate(("view_hero", "view_left_30", "view_right_30")):
                view_dir = root / "renders" / view_id
                view_dir.mkdir(parents=True)
                rect = (28 + index * 2, 44, 108 + index * 2, 92)
                rgb = Image.new("RGB", (128, 128), (34, 36, 40))
                ImageDraw.Draw(rgb).rectangle(rect, fill=(232, 234, 238), outline=(80, 150, 230), width=3)
                rgb_path = view_dir / "rgb.png"
                rgb.save(rgb_path)
                mask = Image.new("L", (128, 128), 0)
                ImageDraw.Draw(mask).rectangle(rect, fill=255)
                mask_path = view_dir / "mask.png"
                mask.convert("RGB").save(mask_path)
                edge = Image.new("L", (128, 128), 0)
                ImageDraw.Draw(edge).rectangle(rect, outline=255, width=3)
                edge_path = view_dir / "edge.png"
                edge.convert("RGB").save(edge_path)
                views.append({"view_id": view_id, "files": {"rgb": str(rgb_path), "mask": str(mask_path), "edge": str(edge_path)}})

                final = root / "final" / f"final_{view_id}.png"
                final.parent.mkdir(parents=True, exist_ok=True)
                final_img = Image.new("RGB", (128, 128), (34, 36, 40))
                final_rect = rect if view_id != "view_left_30" else (70, 12, 124, 54)
                ImageDraw.Draw(final_img).rectangle(final_rect, fill=(232, 234, 238), outline=(80, 150, 230), width=3)
                final_img.save(final)
                final_view_images[view_id] = str(final)

            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": views})
            report = create_white_model_multiview_position_lock_report(
                final_view_images=final_view_images,
                render_manifest=render_manifest,
                output_report=root / "reports" / "white_model_position_lock.json",
                output_image=root / "final" / "white_position_lock_overlay.png",
            )

            self.assertEqual(report["mode"], "multiview")
            self.assertEqual(report["status"], "needs_review")
            self.assertIn("view_left_30", report["failed_views"])
            self.assertEqual(report["view_count"], 3)
            self.assertLess(report["metrics"]["pass_rate"], 1.0)
            self.assertTrue(Path(report["overlay_image"]).exists())

    def test_final_position_retry_plan_uses_white_model_channels_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "renders" / "view_hero"
            view_dir.mkdir(parents=True)
            files = {}
            for channel in ("rgb", "edge", "mask"):
                path = view_dir / f"{channel}.png"
                image = Image.new("RGB", (128, 128), (32, 34, 38))
                ImageDraw.Draw(image).rectangle((24, 40, 104, 92), fill=(230, 232, 236), outline=(90, 160, 240), width=3)
                image.save(path)
                files[channel] = str(path)
            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": [{"view_id": "view_hero", "files": files}]})
            contract = create_white_model_position_contract(
                render_manifest=render_manifest,
                output_report=root / "reports" / "white_model_position_contract.json",
                output_image=root / "final" / "white_position_contract_overlay.png",
                output_views=1,
            )
            self.assertEqual(contract["status"], "pass")
            final_request_path = write_manifest(
                root / "final" / "codex_image2_final_request.json",
                {
                    "type": "codex_image2_final_render_request",
                    "provider": "codex_builtin_image2",
                    "white_model_position_contract": str(root / "reports" / "white_model_position_contract.json"),
                    "requests": [
                        {
                            "view_id": "view_hero",
                            "kind": "final_render",
                            "output_path": str(root / "final" / "final_view_hero.png"),
                            "prompt": "polished product render",
                            "input_images": [
                                {"role": "white_model_rgb_position_lock", "channel": "rgb", "path": files["rgb"]},
                                {"role": "edge_silhouette_lock", "channel": "edge", "path": files["edge"]},
                                {"role": "mask_composition_reference", "channel": "mask", "path": files["mask"]},
                            ],
                            "position_lock_contract": contract["contracts"][0],
                            "position_lock": {"primary_reference_role": "white_model_rgb_position_lock"},
                        }
                    ],
                },
            )
            white_lock_report = {
                "type": "white_model_position_lock",
                "status": "needs_review",
                "final_image": str(root / "final" / "final_view_hero.png"),
                "metrics": {"center_alignment": 0.2, "scale_alignment": 0.4},
                "failure_reasons": ["center_alignment", "scale_alignment"],
            }

            retry = create_final_position_retry_plan(
                workdir=root,
                final_request_path=final_request_path,
                white_lock_report=white_lock_report,
                position_contract_path=root / "reports" / "white_model_position_contract.json",
                output_report=root / "reports" / "final_position_retry_plan.json",
            )

            self.assertEqual(retry["status"], "awaiting_codex_image2_retry")
            retry_request = read_manifest(retry["retry_request"])
            self.assertEqual(retry_request["provider"], "codex_builtin_image2")
            item = retry_request["requests"][0]
            self.assertEqual(item["kind"], "final_render_position_retry")
            self.assertIn("center_alignment", item["prompt"])
            self.assertIn("Binding screen-space target", item["prompt"])
            self.assertEqual(item["edit_target_role"], "white_model_rgb_position_lock")
            self.assertTrue(item["few_shot_position_lock_examples"])
            roles = {entry["role"] for entry in item["input_images"]}
            self.assertIn("white_model_rgb_position_lock", roles)
            self.assertNotIn("appearance_style_reference_only", roles)
            self.assertNotIn("negative_prompt", json.dumps(retry_request, ensure_ascii=False))
            self.assertTrue(Path(retry["codex_image2_handoff"]).exists())
            handoff_text = Path(retry["codex_image2_handoff"]).read_text(encoding="utf-8")
            self.assertIn("Edit target role", handoff_text)
            self.assertIn("Few-shot position-lock examples", handoff_text)

    def test_final_position_retry_plan_includes_all_final_request_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = []
            request_items = []
            for offset, view_id in enumerate(("view_hero", "view_left_30", "view_right_30")):
                view_dir = root / "renders" / view_id
                view_dir.mkdir(parents=True)
                files = {}
                for channel in ("rgb", "edge", "mask"):
                    path = view_dir / f"{channel}.png"
                    image = Image.new("RGB", (128, 128), (32 + offset, 34, 38))
                    ImageDraw.Draw(image).rectangle((24 + offset * 4, 40, 104 + offset * 4, 92), fill=(230, 232, 236), outline=(90, 160, 240), width=3)
                    image.save(path)
                    files[channel] = str(path)
                views.append({"view_id": view_id, "files": files})
                request_items.append(
                    {
                        "view_id": view_id,
                        "kind": "final_render",
                        "output_path": str(root / "final" / f"final_{view_id}.png"),
                        "prompt": f"polished product render {view_id}",
                        "input_images": [
                            {"role": "white_model_rgb_position_lock", "channel": "rgb", "path": files["rgb"]},
                            {"role": "edge_silhouette_lock", "channel": "edge", "path": files["edge"]},
                            {"role": "mask_composition_reference", "channel": "mask", "path": files["mask"]},
                        ],
                        "position_lock": {"primary_reference_role": "white_model_rgb_position_lock"},
                    }
                )
            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": views})
            contract = create_white_model_position_contract(
                render_manifest=render_manifest,
                output_report=root / "reports" / "white_model_position_contract.json",
                output_image=root / "final" / "white_position_contract_overlay.png",
                output_views=3,
            )
            self.assertEqual(contract["status"], "pass")
            contracts = {item["view_id"]: item for item in contract["contracts"]}
            for item in request_items:
                item["position_lock_contract"] = contracts[item["view_id"]]
            final_request_path = write_manifest(
                root / "final" / "codex_image2_final_request.json",
                {
                    "type": "codex_image2_final_render_request",
                    "provider": "codex_builtin_image2",
                    "white_model_position_contract": str(root / "reports" / "white_model_position_contract.json"),
                    "requests": request_items,
                },
            )
            white_lock_report = {
                "type": "white_model_position_lock",
                "status": "needs_review",
                "final_image": str(root / "final" / "final_view_hero.png"),
                "metrics": {"center_alignment": 0.2, "scale_alignment": 0.4},
                "failure_reasons": ["center_alignment", "scale_alignment"],
            }

            retry = create_final_position_retry_plan(
                workdir=root,
                final_request_path=final_request_path,
                white_lock_report=white_lock_report,
                position_contract_path=root / "reports" / "white_model_position_contract.json",
                output_report=root / "reports" / "final_position_retry_plan.json",
            )

            self.assertEqual(retry["status"], "awaiting_codex_image2_retry")
            self.assertEqual(retry["request_count"], 3)
            retry_request = read_manifest(retry["retry_request"])
            self.assertEqual(retry_request["retry_scope"], "all_final_request_views")
            self.assertEqual(retry_request["request_count"], 3)
            self.assertEqual([item["view_id"] for item in retry_request["requests"]], ["view_hero", "view_left_30", "view_right_30"])
            for item in retry_request["requests"]:
                roles = {entry["role"] for entry in item["input_images"]}
                self.assertIn("white_model_rgb_position_lock", roles)
                self.assertIn("edge_silhouette_lock", roles)
                self.assertIn("white-model", item["prompt"])
                self.assertIn("Binding screen-space target", item["prompt"])
                self.assertEqual(item["edit_target_role"], "white_model_rgb_position_lock")
                self.assertTrue(item["few_shot_position_lock_examples"])
            self.assertIn("view_left_30=/path/to/codex-image2-output-2.png", Path(retry["codex_image2_handoff"]).read_text(encoding="utf-8"))
            self.assertNotIn("negative_prompt", json.dumps(retry_request, ensure_ascii=False))

    def test_final_position_retry_plan_synthesizes_request_when_final_image2_request_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "renders" / "view_hero"
            view_dir.mkdir(parents=True)
            files = {}
            for channel in ("rgb", "edge", "mask", "depth", "normal"):
                path = view_dir / f"{channel}.png"
                image = Image.new("RGB", (128, 128), (36, 38, 42))
                ImageDraw.Draw(image).rectangle((26, 42, 108, 92), fill=(232, 234, 238), outline=(80, 150, 230), width=3)
                image.save(path)
                files[channel] = str(path)
            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": [{"view_id": "view_hero", "files": files}]})
            contract = create_white_model_position_contract(
                render_manifest=render_manifest,
                output_report=root / "reports" / "white_model_position_contract.json",
                output_image=root / "final" / "white_position_contract_overlay.png",
                output_views=1,
            )
            self.assertEqual(contract["status"], "pass")
            final_image = root / "final" / "final_view_hero.png"
            final_image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (80, 80, 80)).save(final_image)
            white_lock_report = {
                "type": "white_model_position_lock",
                "status": "needs_review",
                "final_image": str(final_image),
                "metrics": {"center_alignment": 0.1, "scale_alignment": 0.3},
                "failure_reasons": ["center_alignment", "scale_alignment"],
            }

            retry = create_final_position_retry_plan(
                workdir=root,
                final_request_path=root / "final" / "codex_image2_final_request.json",
                white_lock_report=white_lock_report,
                position_contract_path=root / "reports" / "white_model_position_contract.json",
                output_report=root / "reports" / "final_position_retry_plan.json",
            )

            self.assertEqual(retry["status"], "awaiting_codex_image2_retry")
            self.assertEqual(retry["reason"], "white_model_position_lock_needs_review")
            synthesized = read_manifest(root / "final" / "codex_image2_final_request.json")
            self.assertEqual(synthesized["status"], "synthesized_for_position_retry")
            self.assertEqual(synthesized["provider"], "codex_builtin_image2")
            self.assertEqual(synthesized["planning_images_excluded_from_final_inputs"], [])
            retry_request = read_manifest(retry["retry_request"])
            item = retry_request["requests"][0]
            self.assertEqual(item["kind"], "final_render_position_retry")
            roles = {entry["role"] for entry in item["input_images"]}
            self.assertIn("white_model_rgb_position_lock", roles)
            self.assertIn("edge_silhouette_lock", roles)
            self.assertIn("mask_composition_reference", roles)
            self.assertEqual(item["previous_final_image"], str(final_image))
            self.assertIn("center_alignment", item["prompt"])
            self.assertNotIn("negative_prompt", json.dumps(synthesized, ensure_ascii=False))
            self.assertNotIn("negative_prompt", json.dumps(retry_request, ensure_ascii=False))
            self.assertTrue(Path(retry["codex_image2_handoff"]).exists())

    def test_final_position_retry_plan_synthesizes_all_contract_views_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = []
            for offset, view_id in enumerate(("view_hero", "view_left_30", "view_right_30")):
                view_dir = root / "renders" / view_id
                view_dir.mkdir(parents=True)
                files = {}
                for channel in ("rgb", "edge", "mask", "depth", "normal"):
                    path = view_dir / f"{channel}.png"
                    image = Image.new("RGB", (128, 128), (36 + offset, 38, 42))
                    ImageDraw.Draw(image).rectangle((24 + offset * 4, 42, 106 + offset * 4, 92), fill=(232, 234, 238), outline=(80, 150, 230), width=3)
                    image.save(path)
                    files[channel] = str(path)
                views.append({"view_id": view_id, "files": files})
            render_manifest = write_manifest(root / "renders" / "render_manifest.json", {"views": views})
            contract = create_white_model_position_contract(
                render_manifest=render_manifest,
                output_report=root / "reports" / "white_model_position_contract.json",
                output_image=root / "final" / "white_position_contract_overlay.png",
                output_views=3,
            )
            self.assertEqual(contract["contract_count"], 3)
            final_image = root / "final" / "final_view_hero.png"
            final_image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (80, 80, 80)).save(final_image)
            white_lock_report = {
                "type": "white_model_position_lock",
                "status": "needs_review",
                "final_image": str(final_image),
                "metrics": {"center_alignment": 0.1, "scale_alignment": 0.3},
                "failure_reasons": ["center_alignment", "scale_alignment"],
            }

            retry = create_final_position_retry_plan(
                workdir=root,
                final_request_path=root / "final" / "codex_image2_final_request.json",
                white_lock_report=white_lock_report,
                position_contract_path=root / "reports" / "white_model_position_contract.json",
                output_report=root / "reports" / "final_position_retry_plan.json",
            )

            self.assertEqual(retry["request_count"], 3)
            synthesized = read_manifest(root / "final" / "codex_image2_final_request.json")
            self.assertEqual(synthesized["request_count"], 3)
            self.assertEqual([item["view_id"] for item in synthesized["requests"]], ["view_hero", "view_left_30", "view_right_30"])
            output_paths = [Path(item["output_path"]).name for item in synthesized["requests"]]
            self.assertEqual(output_paths, ["final_view_hero.png", "final_view_left_30.png", "final_view_right_30.png"])
            retry_request = read_manifest(retry["retry_request"])
            self.assertEqual(retry_request["request_count"], 3)
            self.assertEqual([item["view_id"] for item in retry_request["requests"]], ["view_hero", "view_left_30", "view_right_30"])
            self.assertNotIn("negative_prompt", json.dumps(synthesized, ensure_ascii=False))
            self.assertNotIn("negative_prompt", json.dumps(retry_request, ensure_ascii=False))

    def test_auto_scene_workflow_mock_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_auto_scene(self._options(root))
            self.assertIn(summary["status"], {"complete", "needs_review"})
            required = [
                "auto_task",
                "scene_plan",
                "concept_image_plan",
                "global_concept",
                "module_plan",
                "module_asset_manifest",
                "module_mesh_sanity",
                "module_assets_index",
                "module_references_contact_sheet",
                "scene_assembly",
                "final_scene_manifest",
                "scene_model_path",
                "assembly_report",
                "scene_preview",
                "camera_plan",
                "render_manifest",
                "white_model_position_contract",
                "white_position_contract_overlay",
                "white_render",
                "white_channel_contact_sheet",
                "module_scores",
                "structure_scores",
                "multiview_score",
                "agent_report",
                "tool_calls",
                "concept_final_comparison",
                "concept_vs_final",
                "white_model_position_lock",
                "white_position_lock_overlay",
                "final_position_retry_plan",
                "final_image",
                "contact_sheet",
                "stages",
            ]
            for key in required:
                self.assertTrue(Path(summary[key]).exists(), key)
            self.assertTrue(Path(summary["final_view_images"]["view_left_30"]).exists())
            self.assertTrue(Path(summary["final_view_images"]["view_right_30"]).exists())
            self.assertTrue(Path(summary["concept_review"]).exists())
            self.assertTrue(Path(summary["module_prompt_info"]).exists())
            self.assertTrue(Path(summary["module_reference_review"]).exists())
            task = read_manifest(summary["auto_task"])
            self.assertEqual(task["mode"], "modular_scene_agent")
            self.assertEqual(task["planner"], "mock_model_brain")
            concept_generation = read_manifest(Path(summary["global_concept"]).with_name("generation_manifest.json"))
            self.assertEqual(concept_generation["created_by"], "mock_image2_generation")
            self.assertTrue(Path(concept_generation["prompt_file"]).exists())
            module_review = read_manifest(summary["module_reference_review"])
            self.assertEqual(module_review["status"], "pass")
            white_lock = read_manifest(summary["white_model_position_lock"])
            self.assertIn(white_lock["status"], {"pass", "needs_review"})
            self.assertIn("total", white_lock["metrics"])
            self.assertEqual(summary["capabilities"]["white_model_position_lock"]["status"], white_lock["status"])
            stages = read_manifest(summary["stages"])
            self.assertEqual(stages["type"], "auto_scene_stages")
            self.assertEqual(stages["stage_count"], len(stages["stages"]))
            by_id = {stage["id"]: stage for stage in stages["stages"]}
            self.assertEqual(set(by_id), set([
                "understand",
                "concept",
                "decompose",
                "module_reference",
                "module_3d",
                "module_check",
                "layout",
                "scene_preview",
                "camera",
                "render",
                "agent",
                "score",
                "consistency",
                "package",
                "complete",
            ]))
            for stage in stages["stages"]:
                self.assertIn(stage["status"], {"complete", "needs_review", "pending", "failed", "awaiting_external_imagegen"})
                self.assertIn("artifacts", stage)
                self.assertIn("warnings", stage)
                self.assertIn("error", stage)
                self.assertIn("retry_count", stage)
                self.assertTrue(stage["message"])
            self.assertIn("global_concept", by_id["concept"]["artifacts"])
            self.assertIn("module_assets_index", by_id["module_reference"]["artifacts"])
            self.assertIn("module_references_contact_sheet", by_id["module_reference"]["artifacts"])
            self.assertIn("module_mesh_sanity", by_id["module_check"]["artifacts"])
            self.assertIn("assembly_report", by_id["scene_preview"]["artifacts"])
            self.assertIn("render_manifest", by_id["render"]["artifacts"])
            self.assertIn("white_render", by_id["render"]["artifacts"])
            self.assertIn("white_channel_contact_sheet", by_id["render"]["artifacts"])
            self.assertIn("white_model_position_contract", by_id["agent"]["artifacts"])
            self.assertIn("white_position_contract_overlay", by_id["agent"]["artifacts"])
            self.assertIn("structure_scores", by_id["score"]["artifacts"])
            self.assertIn("multiview_score", by_id["score"]["artifacts"])
            self.assertIn("multiview_score", by_id["consistency"]["artifacts"])
            self.assertIn("white_model_position_lock", by_id["package"]["artifacts"])
            self.assertIn("final_position_retry_plan", by_id["package"]["artifacts"])
            module_plan = read_manifest(summary["module_plan"])
            self.assertEqual(module_plan["modules"][0]["module_id"], "primary_subject")
            modules = read_manifest(summary["module_asset_manifest"])["modules"]
            mesh_sanity = read_manifest(summary["module_mesh_sanity"])
            self.assertEqual(mesh_sanity["type"], "module_mesh_sanity")
            self.assertEqual(mesh_sanity["status"], "pass")
            self.assertEqual(summary["capabilities"]["module_mesh_sanity"]["status"], "pass")
            position_contract = read_manifest(summary["white_model_position_contract"])
            self.assertEqual(position_contract["type"], "white_model_position_contract")
            self.assertEqual(position_contract["status"], "pass")
            self.assertGreaterEqual(position_contract["contract_count"], 1)
            self.assertTrue(Path(summary["white_position_contract_overlay"]).exists())
            retry_plan = read_manifest(summary["final_position_retry_plan"])
            self.assertEqual(retry_plan["type"], "final_position_retry_plan")
            self.assertIn(retry_plan["status"], {"not_needed", "not_applicable", "awaiting_codex_image2_retry"})
            self.assertGreaterEqual(len(modules), 3)
            module_index = read_manifest(summary["module_assets_index"])
            self.assertEqual(module_index["type"], "module_assets_index")
            self.assertEqual(module_index["module_count"], len(modules))
            self.assertTrue(Path(module_index["module_reference_contact_sheet"]).exists())
            indexed_by_id = {item["module_id"]: item for item in module_index["modules"]}
            self.assertEqual(set(indexed_by_id), {item["module_id"] for item in modules})
            for item in module_index["modules"]:
                self.assertTrue(Path(item["reference_image"]).exists(), item["module_id"])
                self.assertTrue(Path(item["model_path"]).exists(), item["module_id"])
                self.assertTrue(item["reference_exists"], item["module_id"])
                self.assertTrue(item["model_exists"], item["module_id"])
            hero_asset = next(item for item in modules if item["module_id"] == module_plan["modules"][0]["module_id"])
            self.assertGreaterEqual(hero_asset["sanity"]["vertices"], 8)
            self.assertEqual(hero_asset["sanity"]["status"], "pass")
            self.assertEqual(read_manifest(hero_asset["metadata"])["created_by"], "procedural_module_proxy_v3_axis_corrected")
            hero_reference = read_manifest(root / "modules" / hero_asset["module_id"] / "reference_manifest.json")
            self.assertEqual(hero_reference["review_status"], "pass")
            self.assertEqual(hero_reference["created_by"], "mock_image2_generation")
            self.assertTrue(Path(hero_reference["prompt_file"]).exists())
            assembly_report = read_manifest(Path(summary["scene_model_path"]).with_name("assembly_report.json"))
            self.assertEqual(summary["assembly_report"], str(Path(summary["scene_model_path"]).with_name("assembly_report.json").resolve()))
            final_scene_manifest = read_manifest(summary["final_scene_manifest"])
            self.assertEqual(final_scene_manifest["assembly_report"], summary["assembly_report"])
            self.assertEqual(assembly_report["geometry_generator"], "hybrid_scene_proxy_v4_external_modules")
            self.assertEqual(assembly_report["coordinate_export"], "blender_z_up_to_gltf_y_up")
            self.assertGreater(assembly_report["vertices"], 8)
            render_manifest = read_manifest(summary["render_manifest"])
            self.assertEqual(render_manifest["scene_model_path"], summary["scene_model_path"])
            self.assertIn("module_ids_visible", render_manifest["views"][0])
            self.assertEqual(summary["white_render"], render_manifest["views"][0]["files"]["rgb"])
            self.assertTrue(Path(summary["white_render"]).exists())
            self.assertTrue(Path(summary["white_channel_contact_sheet"]).exists())
            self.assertEqual(
                summary["reference_policy"]["final_ai_inputs"],
                [
                    "render_manifest.rgb",
                    "render_manifest.edge",
                    "render_manifest.depth",
                    "render_manifest.normal",
                    "render_manifest.mask",
                    "render_manifest.skeleton_from_mask",
                ],
            )

    def test_camera_preview_scoring_uses_concept_composition_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept.png"
            matching = root / "matching.png"
            shifted = root / "shifted.png"

            for path, rect in (
                (concept, (120, 190, 285, 320)),
                (matching, (120, 190, 285, 320)),
                (shifted, (290, 190, 455, 320)),
            ):
                image = Image.new("RGB", (512, 512), (18, 22, 28))
                draw = ImageDraw.Draw(image)
                draw.rectangle(rect, fill=(236, 240, 245), outline=(80, 180, 255), width=4)
                draw.rectangle((rect[0] + 28, rect[1] + 42, rect[2] - 28, rect[3] - 32), outline=(20, 45, 70), width=3)
                image.save(path)

            target = _extract_concept_camera_target(concept)
            self.assertEqual(target["status"], "pass")
            match_score = _score_camera_preview_image(matching, concept_target=target)
            shifted_score = _score_camera_preview_image(shifted, concept_target=target)
            self.assertGreater(match_score["concept_center_score"], shifted_score["concept_center_score"])
            self.assertGreater(match_score["score"], shifted_score["score"])

    def test_select_scene_camera_records_concept_target_when_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept = root / "concept.png"
            image = Image.new("RGB", (256, 256), (20, 22, 26))
            ImageDraw.Draw(image).rectangle((70, 96, 190, 170), fill=(230, 234, 240))
            image.save(concept)
            scene_model = root / "scene.glb"
            scene_model.write_bytes(b"glTF")

            selection = select_scene_camera(
                root,
                {"scene_model_path": str(scene_model)},
                {"views": [{"view_id": "view_hero", "azimuth_deg": 310.0}]},
                views=3,
                config={},
                render_backend="procedural",
                dry_run=True,
                concept_image=concept,
            )

            report = read_manifest(selection["report_path"])
            self.assertEqual(report["status"], "skipped")
            self.assertEqual(report["concept_camera_target"]["status"], "pass")

    def test_external_hero_model_is_preserved_for_scene_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, _ = call_model_scene_planner({}, self._options(root))
            source_model = root / "source_hero.glb"
            source_model.parent.mkdir(parents=True, exist_ok=True)
            # Reuse the procedural writer to create a valid GLB fixture; the source itself is treated as external.
            from local3dai.auto_scene import write_module_proxy_glb

            hero = next(item for item in plan["module_plan"]["modules"] if item["role"] == "hero_object")
            write_module_proxy_glb(source_model, hero)
            assets = generate_module_assets(root, plan["module_plan"], allow_fallback=True, hero_model_path=source_model)
            hero_asset = next(item for item in assets["modules"] if item["module_id"] == hero["module_id"])
            metadata = read_manifest(hero_asset["metadata"])
            self.assertEqual(metadata["created_by"], "external_hero_model_glb")
            self.assertEqual(metadata["source_model_path"], str(source_model))
            scene_assembly = plan_scene_layout(plan["scene_plan"], plan["module_plan"], assets)
            scene_outputs = assemble_scene(root, scene_assembly)
            assembly_report = read_manifest(scene_outputs["assembly_report"])
            external_hero = next(item for item in assembly_report["external_modules"] if item["module_id"] == hero["module_id"])
            self.assertEqual(external_hero["source_model_path"], hero_asset["model_path"])
            self.assertEqual(
                external_hero["axis_transform"]["coordinate_normalization"],
                "hunyuan_or_external_glb_to_blender_z_up_scene_axes",
            )
            self.assertTrue(external_hero["axis_transform"]["axis_mapping"])

    def test_module_assets_use_hunyuan_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "modules" / "hero_car" / "reference.png"
            reference.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (240, 240, 240)).save(reference)
            module_plan = {
                "modules": [
                    {
                        "module_id": "hero_car",
                        "name": "hero car",
                        "role": "supporting_object",
                        "category": "vehicle",
                        "generate_3d": True,
                        "expected_real_world_size": {"width": 2.0, "depth": 4.8, "height": 1.2, "unit": "meters"},
                    }
                ]
            }
            seen_overrides: dict[str, object] = {}

            def fake_hunyuan_shape(**kwargs: object) -> Path:
                from local3dai.auto_scene import write_module_proxy_glb

                seen_overrides.update(dict(kwargs.get("shape_overrides") or {}))
                output_model = Path(kwargs["output_model"])
                write_module_proxy_glb(output_model, module_plan["modules"][0])
                write_manifest(
                    Path(kwargs["metadata"]),
                    {"backend": "fake_hunyuan", "mesh_sanity": {"status": "needs_review", "flags": ["test_flag"]}},
                )
                return output_model

            options = self._options(root)
            options.dry_run = False
            options.backend = None
            config = {
                "system": {"python": ".venv/bin/python"},
                "models": {
                    "hunyuan3d_2_1_shape": {
                        "enabled": True,
                        "default_profile": "high",
                        "profiles": {"high": {"steps": 40, "guidance_scale": 5.0, "octree_resolution": 512, "num_chunks": 20000}},
                    }
                },
            }
            with patch("local3dai.workflow._run_hunyuan_shape", side_effect=fake_hunyuan_shape):
                assets = generate_module_assets(root, module_plan, allow_fallback=True, config=config, options=options)
            asset = assets["modules"][0]
            metadata = read_manifest(asset["metadata"])
            self.assertEqual(metadata["created_by"], "hunyuan3d_2_1_shape_from_reviewed_reference")
            self.assertIn("hunyuan_shape_metadata.json", metadata["hunyuan_metadata"])
            self.assertEqual(asset["sanity"]["source_reference_image"], str(reference))
            self.assertEqual(asset["sanity"]["status"], "needs_review")
            self.assertEqual(asset["sanity"]["hunyuan_mesh_sanity"]["flags"], ["test_flag"])
            self.assertEqual(asset["sanity"]["hunyuan_profile"], "high")
            self.assertEqual(seen_overrides["steps"], 40)
            self.assertEqual(seen_overrides["octree_resolution"], 512)

    def test_module_assets_flag_platform_mesh_that_is_not_low_slab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "modules" / "display_platform" / "reference.png"
            reference.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (180, 180, 180)).save(reference)
            module_plan = {
                "modules": [
                    {
                        "module_id": "display_platform",
                        "name": "low display platform",
                        "role": "supporting_object",
                        "category": "stage_prop",
                        "generate_3d": True,
                        "expected_real_world_size": {"width": 5.0, "depth": 4.0, "height": 0.25, "unit": "meters"},
                    }
                ]
            }

            def fake_hunyuan_shape(**kwargs: object) -> Path:
                from local3dai.auto_scene import write_module_proxy_glb

                output_model = Path(kwargs["output_model"])
                write_module_proxy_glb(
                    output_model,
                    {
                        "module_id": "generic_tall_block",
                        "expected_real_world_size": {"width": 1.0, "depth": 1.0, "height": 3.0, "unit": "meters"},
                    },
                )
                write_manifest(Path(kwargs["metadata"]), {"backend": "fake_hunyuan", "mesh_sanity": {"status": "pass"}})
                return output_model

            options = self._options(root)
            options.dry_run = False
            options.backend = None
            config = {
                "models": {
                    "hunyuan3d_2_1_shape": {
                        "enabled": True,
                        "default_profile": "high",
                        "profiles": {"high": {"steps": 40, "octree_resolution": 512}},
                    }
                },
            }
            with patch("local3dai.workflow._run_hunyuan_shape", side_effect=fake_hunyuan_shape):
                assets = generate_module_assets(root, module_plan, allow_fallback=True, config=config, options=options)

            asset = assets["modules"][0]
            self.assertEqual(assets["status"], "needs_review")
            self.assertEqual(assets["quality_issues"][0]["module_id"], "display_platform")
            semantic = asset["sanity"]["semantic_mesh_sanity"]
            self.assertEqual(semantic["status"], "needs_review")
            codes = {flag["code"] for flag in semantic["flags"]}
            self.assertIn("platform_not_low_slab", codes)


if __name__ == "__main__":
    unittest.main()
