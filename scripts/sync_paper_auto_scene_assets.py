from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

def _resolve(path: str | Path, base: Path) -> Path:
    item = Path(path)
    if not item.is_absolute():
        item = base / item
    return item


def _copy(path: Path, output: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output)
    return output


def _render_manifest_rgb(workdir: Path, view_id: str) -> Path:
    manifest_path = workdir / "renders" / "render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aliases = {view_id}
    if view_id == "view_hero":
        aliases.add("view_locked")
    for view in manifest.get("views", []):
        if str(view.get("view_id", "")) in aliases:
            rgb = view.get("files", {}).get("rgb", "")
            if rgb:
                return _resolve(rgb, workdir)
    raise FileNotFoundError(f"missing RGB render for {view_id} in {manifest_path}")


def _sync_one_destination(workdir: Path, destination: Path) -> dict[str, str]:
    white_hero = _render_manifest_rgb(workdir, "view_hero")
    final_dir = workdir / "final"
    sources = {
        "module_references_contact": final_dir / "module_references_contact_sheet.png",
        "scene_preview": white_hero,
        "render_channels_view_hero": final_dir / "white_channels_contact_sheet.png",
        "final_contact_sheet": final_dir / "contact_sheet.png",
        "white_vs_final": final_dir / "white_vs_final.png",
        "concept_vs_final": final_dir / "concept_vs_final.png",
        "white_model_view_hero": white_hero,
        "final_view_hero": final_dir / "final_view_hero.png",
    }
    written = {name: str(_copy(source, destination / f"{name}.png")) for name, source in sources.items()}

    overlay = workdir / "final" / "white_position_lock_overlay.png"
    if overlay.exists():
        written["white_position_lock_overlay"] = str(_copy(overlay, destination / "white_position_lock_overlay.png"))

    return written


def sync_assets(workdir: Path, destinations: list[Path]) -> dict[str, object]:
    if not workdir.exists():
        raise FileNotFoundError(workdir)
    result = {
        "type": "paper_auto_scene_asset_sync",
        "workdir": str(workdir.resolve()),
        "destinations": [],
    }
    for destination in destinations:
        result["destinations"].append(
            {
                "path": str(destination.resolve()),
                "assets": _sync_one_destination(workdir, destination),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Auto Scene assets into paper asset directories.")
    parser.add_argument("--workdir", required=True, help="Auto Scene workdir containing renders/ and final/.")
    parser.add_argument(
        "--asset-dir",
        action="append",
        default=[],
        help="Destination paper_assets directory. May be passed more than once.",
    )
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    destinations = [Path(item) for item in args.asset_dir] or [Path("docs/paper_assets"), Path("reports/paper_assets")]
    report = sync_assets(Path(args.workdir), destinations)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
