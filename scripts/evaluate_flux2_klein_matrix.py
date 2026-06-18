#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from local3dai.agent import AgentRunOptions, run_agent_render
from local3dai.ai.backends import build_backend
from local3dai.config import load_config
from local3dai.manifest import read_manifest, write_manifest
from local3dai.scoring import score_candidates


DEFAULT_MODES: list[tuple[str, list[str], bool]] = [
    ("text_only", [], False),
    ("rgb", ["rgb"], False),
    ("rgb_edge", ["rgb", "edge"], False),
    ("rgb_depth", ["rgb", "depth"], False),
    ("rgb_edge_depth_normal_mask_skeleton", ["rgb", "edge", "depth", "normal", "mask", "skeleton"], False),
    ("full_refs_geometry_lock", ["rgb", "edge", "depth", "normal", "mask", "skeleton"], True),
]


def _model_ref(model: dict[str, Any]) -> str:
    return model.get("local_path") or model.get("model_path") or model.get("model_id") or ""


def _single_view_manifest(source_manifest: Path, output: Path, view_id: str) -> Path:
    manifest = read_manifest(source_manifest)
    selected = None
    for view in manifest.get("views", []):
        if view.get("view_id") == view_id:
            selected = view
            break
    if selected is None:
        selected = manifest["views"][0]
    single = {
        **manifest,
        "views": [selected],
        "view_graph": [],
    }
    return write_manifest(output, single)


def _best_from_report(report_path: Path) -> dict[str, Any]:
    report = read_manifest(report_path)
    ranked = report.get("ranked") or []
    if not ranked:
        return {}
    best = ranked[0]
    return {
        "file": best.get("ranked_copy") or best.get("file", ""),
        "scores": best.get("scores", {}),
        "candidate_id": best.get("candidate_id", ""),
        "view_id": best.get("view_id", ""),
    }


def _make_contact(items: list[dict[str, str]], output: Path) -> Path:
    if not items:
        return output
    tile = 320
    pad = 12
    label_h = 34
    width = pad + len(items) * (tile + pad)
    height = tile + label_h + pad * 2
    canvas = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["file"]).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        x = pad + index * (tile + pad)
        canvas.paste(image, (x, pad + label_h))
        draw.text((x, pad + 10), item["label"], fill=(242, 242, 242))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    model_key = args.model_key or config["ai"].get("default_model_key", "flux2_klein_4b")
    model = dict(config["models"][model_key])
    backend_name = model.get("backend", config["ai"].get("default_backend", "flux2-klein"))
    backend = build_backend(backend_name)
    single_manifest = _single_view_manifest(Path(args.manifest), root / "single_view_manifest.json", args.view_id)
    score_cfg = config["score"]

    entries: list[dict[str, Any]] = []
    contact_items: list[dict[str, str]] = []
    for mode_name, channels, geometry_lock in DEFAULT_MODES:
        mode_dir = root / mode_name
        ai_manifest = backend.generate(
            single_manifest,
            mode_dir / "candidates",
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or model.get("negative_prompt", ""),
            candidates_per_view=1,
            seed=args.seed,
            model_ref=_model_ref(model),
            model_config={**model, "reference_channels": channels, "geometry_lock": geometry_lock},
            device=args.device or config["ai"].get("device", "cuda:0"),
            dtype=args.dtype or model.get("dtype") or config["ai"].get("dtype", "bfloat16"),
            steps=args.steps or int(model.get("steps", config["ai"].get("steps", 4))),
            guidance_scale=args.guidance_scale if args.guidance_scale is not None else float(model.get("guidance_scale", config["ai"].get("guidance_scale", 1.0))),
            width=args.width or int(model.get("width", 1024)),
            height=args.height or int(model.get("height", 1024)),
            reference_channels=channels,
            geometry_lock=geometry_lock,
        )
        report = score_candidates(
            render_manifest_path=single_manifest,
            ai_manifest_path=ai_manifest,
            output_dir=mode_dir / "score",
            copy_top_k=1,
            version=str(score_cfg.get("version", "structure_v2")),
            structure_weights=dict(score_cfg.get("structure_v2", {})),
        )
        best = _best_from_report(report)
        entries.append(
            {
                "mode": mode_name,
                "reference_channels": channels,
                "geometry_lock": geometry_lock,
                "ai_manifest": str(ai_manifest),
                "score_report": str(report),
                "best": best,
            }
        )
        if best.get("file"):
            contact_items.append({"label": mode_name, "file": best["file"]})

    agent_dir = root / "multiview_agent"
    agent_summary = run_agent_render(
        AgentRunOptions(
            input_renders=Path(args.manifest),
            output_dir=agent_dir,
            prompt=args.prompt,
            config_path=Path(args.config),
            model_key=model_key,
            backend=backend_name,
            target_view=args.view_id,
            max_generations=args.agent_generations,
            seed=args.seed,
            expand_views=True,
            default_reference_channels=("rgb", "edge", "depth", "normal", "mask", "skeleton"),
            pass_threshold=float(config.get("agent", {}).get("pass_threshold", 0.62)),
            negative_prompt=args.negative_prompt,
            device=args.device,
            dtype=args.dtype or model.get("dtype") or config["ai"].get("dtype", "bfloat16"),
            steps=args.steps,
            width=args.width,
            height=args.height,
        )
    )
    if agent_summary.get("final_image"):
        contact_items.append({"label": "multiview_agent", "file": agent_summary["final_image"]})

    contact = _make_contact(contact_items, root / "flux2_matrix_contact_sheet.png")
    summary = {
        "type": "flux2_klein_matrix",
        "model_key": model_key,
        "backend": backend_name,
        "source_manifest": str(Path(args.manifest)),
        "single_view_manifest": str(single_manifest),
        "prompt": args.prompt,
        "entries": entries,
        "multiview_agent": agent_summary,
        "contact_sheet": str(contact),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = root / "flux2_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if agent_summary.get("multiview_contact_sheet"):
        shutil.copy2(agent_summary["multiview_contact_sheet"], root / "flux2_multiview_contact_sheet.png")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Flux2 Klein from text-only to multi-reference and multiview Agent.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--model-key", default="")
    parser.add_argument("--view-id", default="view_locked")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument("--agent-generations", type=int, default=9)
    args = parser.parse_args()
    summary = run_matrix(args)
    print("Wrote Flux2 matrix summary:", Path(args.output) / "flux2_matrix_summary.json")
    print("Contact sheet:", summary["contact_sheet"])
    print("Multiview report:", summary["multiview_agent"]["agent_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
