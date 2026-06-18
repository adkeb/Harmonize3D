#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from local3dai.ai.geometry import (
    mesh_adaptive_lock_render,
    mesh_detail_lock_render,
    mesh_position_lock_render,
    mesh_quality_lock_render,
)
from local3dai.scoring_v2 import score_structure_v2


LOCKS: dict[str, Callable[..., Path]] = {
    "position": mesh_position_lock_render,
    "detail": mesh_detail_lock_render,
    "adaptive": mesh_adaptive_lock_render,
    "quality": mesh_quality_lock_render,
}


def _make_contact(items: list[dict[str, str]], output: Path, *, tile: int = 320, cols: int = 2) -> Path:
    if not items:
        return output
    pad = 12
    label_h = 34
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (pad + cols * (tile + pad), pad + rows * (tile + label_h + pad)), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["file"]).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = pad + (index % cols) * (tile + pad)
        y = pad + (index // cols) * (tile + label_h + pad)
        draw.text((x, y + 8), item["label"], fill=(242, 242, 242))
        canvas.paste(image, (x, y + label_h + (tile - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _report_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(source.glob("*/score/report.json"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = LOCKS[args.mode]
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    detail_reference = Path(args.detail_reference) if args.detail_reference else None

    items: list[dict[str, Any]] = []
    for report_path in _report_paths(source):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        view_id = report_path.parts[-3] if report_path.name == "report.json" else report_path.stem
        view_dir = output / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        for ranked in report.get("ranked", []):
            candidate = ranked.get("candidate") or {}
            direct = candidate.get("direct_file")
            files = candidate.get("source_files") or {}
            if not direct or not files:
                continue
            target = view_dir / f"{ranked['candidate_id']}_{args.mode}.png"
            kwargs: dict[str, Any] = {
                "source_rgb": files["rgb"],
                "source_mask": files["mask"],
                "source_edge": files["edge"],
                "ai_image": direct,
                "output": target,
            }
            if args.mode in {"detail", "adaptive", "quality"}:
                kwargs["detail_reference"] = detail_reference
            lock(**kwargs)
            scores = score_structure_v2(candidate_path=target, source_files=files)
            items.append(
                {
                    "view_id": view_id,
                    "candidate_id": ranked["candidate_id"],
                    "direct_file": direct,
                    "file": str(target),
                    "scores": scores,
                    "source_scores": ranked.get("scores", {}),
                }
            )

    by_view: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_view.setdefault(item["view_id"], []).append(item)
    for view_items in by_view.values():
        view_items.sort(key=lambda item: float(item["scores"]["total"]), reverse=True)

    contact_items: list[dict[str, str]] = []
    for view_id in sorted(by_view):
        best = by_view[view_id][0]
        contact_items.append({"label": f"{view_id} {args.mode} {best['scores']['total']:.3f}", "file": best["file"]})
        contact_items.append({"label": f"{view_id} direct", "file": best["direct_file"]})
    contact = _make_contact(contact_items, output / f"{args.mode}_preview_contact.png")

    summary = {
        "type": "mesh_lock_preview_from_reports",
        "mode": args.mode,
        "source": str(source),
        "detail_reference": str(detail_reference) if detail_reference else "",
        "contact_sheet": str(contact),
        "items": items,
    }
    report_path = output / f"{args.mode}_preview_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["report"] = str(report_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a mesh lock mode to existing direct candidates from score reports.")
    parser.add_argument("--source", required=True, help="A score report or a view_candidates directory containing */score/report.json.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=sorted(LOCKS), default="quality")
    parser.add_argument("--detail-reference", default="")
    args = parser.parse_args()
    report = run(args)
    print("Wrote report:", report["report"])
    print("Contact:", report["contact_sheet"])
    by_view: dict[str, list[dict[str, Any]]] = {}
    for item in report["items"]:
        by_view.setdefault(item["view_id"], []).append(item)
    for view_id in sorted(by_view):
        best = sorted(by_view[view_id], key=lambda item: float(item["scores"]["total"]), reverse=True)[0]
        scores = best["scores"]
        print(
            view_id,
            best["candidate_id"],
            "total",
            scores["total"],
            "iou",
            scores["silhouette_iou"],
            "edge",
            scores["edge_chamfer_score"],
            "added",
            scores["added_part_penalty"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
