from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .config import load_config
from .workflow import WorkflowOptions, run_workflow


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
OUTPUTS_ROOT = ROOT / "outputs"
UPLOAD_ROOT = OUTPUTS_ROOT / "web-uploads"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _is_safe_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _file_url(path: str | Path) -> str:
    return f"/api/file?path={str(path)}"


def _model_status(config: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for key, model in config.get("models", {}).items():
        paths: list[str] = []
        for path_key in ("local_path", "base_model", "canny_controlnet", "depth_controlnet", "repo_root", "model_path"):
            value = model.get(path_key)
            if value:
                paths.append(str(value))
        status[key] = {
            "backend": model.get("backend", ""),
            "enabled": bool(model.get("enabled", False)),
            "paths": [{"path": path, "exists": Path(path).exists()} for path in paths],
        }
    return status


def _latest_artifacts() -> dict[str, str]:
    candidates = [
        OUTPUTS_ROOT / "hunyuan_shape_sd_render" / "white_vs_geometry_locked_render.png",
        OUTPUTS_ROOT / "hunyuan_shape_sd_render" / "final_geometry_locked_render.png",
        OUTPUTS_ROOT / "hunyuan_shape_sd_render" / "white_model_reference.png",
    ]
    result: dict[str, str] = {}
    labels = ("comparison_image", "final_image", "white_image")
    for label, path in zip(labels, candidates):
        if path.exists():
            result[label] = str(path)
            result[f"{label}_url"] = _file_url(path)
    web_runs = sorted((OUTPUTS_ROOT / "web-runs").glob("run-*/run_summary.json")) if (OUTPUTS_ROOT / "web-runs").exists() else []
    if web_runs:
        try:
            latest = json.loads(web_runs[-1].read_text(encoding="utf-8"))
            for key in ("comparison_image", "final_image", "white_image"):
                if latest.get(key) and Path(latest[key]).exists():
                    result[key] = latest[key]
                    result[f"{key}_url"] = _file_url(latest[key])
        except Exception:
            pass
    return result


def _job_snapshot(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))
        job["logs"] = list(job.get("logs", []))
        return job


def _append_log(job_id: str, stage: str, message: str, progress: float) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["stage"] = stage
        job["progress"] = progress
        job.setdefault("logs", []).append(
            {
                "time": time.strftime("%H:%M:%S"),
                "stage": stage,
                "message": message,
                "progress": progress,
            }
        )


def _run_job(job_id: str, options: WorkflowOptions) -> None:
    try:
        summary = run_workflow(options, progress=lambda stage, message, fraction: _append_log(job_id, stage, message, fraction))
        artifacts = {
            key: summary[key]
            for key in ("white_image", "final_image", "comparison_image", "model_path", "render_manifest", "ai_manifest", "score_report")
            if summary.get(key)
        }
        artifacts.update({f"{key}_url": _file_url(value) for key, value in artifacts.items() if Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}})
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "complete",
                    "progress": 1.0,
                    "summary": summary,
                    "artifacts": artifacts,
                    "finished_at": time.time(),
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_at": time.time(),
                }
            )


def _start_job(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    workdir = OUTPUTS_ROOT / "web-runs" / run_id
    options = WorkflowOptions(
        prompt=payload.get("prompt") or "faceted black obsidian sci fi artifact, cyan magenta studio lighting",
        workdir=workdir,
        source_mode=payload.get("source_mode") or "existing_mesh",
        model_path=Path(payload["model_path"]) if payload.get("model_path") else None,
        reference_image=Path(payload["reference_image"]) if payload.get("reference_image") else None,
        views=int(payload.get("views") or 1),
        render_resolution=int(payload.get("render_resolution") or 1536),
        render_samples=int(payload.get("render_samples") or 64),
        ai_width=int(payload.get("ai_width") or 1536),
        ai_height=int(payload.get("ai_height") or 1536),
        candidates=int(payload.get("candidates") or 1),
        seed=int(payload.get("seed") or 20260610),
        steps=int(payload.get("steps") or 42),
        guidance_scale=float(payload.get("guidance_scale") or 8.0),
        canny_scale=float(payload.get("canny_scale") or 2.85),
        depth_scale=float(payload.get("depth_scale") or 0.55),
        geometry_lock=bool(payload.get("geometry_lock", True)),
        negative_prompt=payload.get("negative_prompt") or "",
        model_key=payload.get("model_key") or "sdxl_controlnet_geometry",
    )
    with JOBS_LOCK:
        JOBS[run_id] = {
            "id": run_id,
            "status": "running",
            "stage": "queued",
            "progress": 0.0,
            "workdir": str(workdir),
            "logs": [],
            "created_at": time.time(),
        }
    thread = threading.Thread(target=_run_job, args=(run_id, options), daemon=True)
    thread.start()
    return _job_snapshot(run_id)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "Local3DAIWeb/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def _send_file_response(self, path: Path, *, include_body: bool = True) -> None:
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/file":
            params = parse_qs(parsed.query)
            requested = Path(params.get("path", [""])[0])
            if not requested.is_absolute():
                requested = ROOT / requested
            requested = requested.resolve()
            if not _is_safe_path(requested) or not requested.exists() or not requested.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            self._send_file_response(requested, include_body=False)
            return
        static_path = WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
        static_path = static_path.resolve()
        if not _is_safe_path(static_path) or not static_path.exists() or not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self._send_file_response(static_path, include_body=False)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/config":
                config = load_config(ROOT / "configs/local.json")
                _json_response(
                    self,
                    {
                        "workspace": str(ROOT),
                        "outputs": str(OUTPUTS_ROOT),
                        "model_status": _model_status(config),
                        "defaults": {
                            "prompt": "same exact 3d model silhouette, faceted black obsidian bird mask, long central beak preserved, large hollow open side slots preserved, side handles preserved, angular top facets preserved, dark reflective ceramic glass, subtle cyan magenta glowing seams, studio render",
                            "negative_prompt": config["models"]["sdxl_controlnet_geometry"].get("negative_prompt", ""),
                            "render_resolution": config["render"]["resolution"],
                            "ai_width": config["models"]["sdxl_controlnet_geometry"].get("width", 1536),
                            "ai_height": config["models"]["sdxl_controlnet_geometry"].get("height", 1536),
                            "steps": config["models"]["sdxl_controlnet_geometry"].get("steps", 42),
                            "canny_scale": config["models"]["sdxl_controlnet_geometry"].get("canny_scale", 2.85),
                            "depth_scale": config["models"]["sdxl_controlnet_geometry"].get("depth_scale", 0.55),
                        },
                        "latest": _latest_artifacts(),
                    },
                )
                return
            if path == "/api/jobs":
                with JOBS_LOCK:
                    jobs = sorted((dict(job) for job in JOBS.values()), key=lambda item: item.get("created_at", 0), reverse=True)
                _json_response(self, {"jobs": jobs})
                return
            if path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = _job_snapshot(job_id)
                if not job:
                    _json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                    return
                _json_response(self, job)
                return
            if path == "/api/file":
                params = parse_qs(parsed.query)
                requested = Path(params.get("path", [""])[0])
                if not requested.is_absolute():
                    requested = ROOT / requested
                requested = requested.resolve()
                if not _is_safe_path(requested) or not requested.exists() or not requested.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                    return
                self._send_file_response(requested)
                return

            static_path = WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
            static_path = static_path.resolve()
            if not _is_safe_path(static_path) or not static_path.exists() or not static_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send_file_response(static_path)
        except Exception as exc:
            _json_response(self, {"error": str(exc), "traceback": traceback.format_exc()}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/run":
                payload = _read_json(self)
                _json_response(self, _start_job(payload), HTTPStatus.ACCEPTED)
                return
            if path == "/api/upload":
                ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
                if ctype != "multipart/form-data":
                    _json_response(self, {"error": "multipart/form-data required"}, HTTPStatus.BAD_REQUEST)
                    return
                pdict["boundary"] = bytes(pdict["boundary"], "utf-8")
                pdict["CONTENT-LENGTH"] = int(self.headers.get("Content-Length", "0"))
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"}, keep_blank_values=True)
                item = form["file"] if "file" in form else None
                if item is None or not getattr(item, "filename", ""):
                    _json_response(self, {"error": "file field is required"}, HTTPStatus.BAD_REQUEST)
                    return
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                safe_name = Path(item.filename).name.replace(" ", "_")
                target = UPLOAD_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_name}"
                with target.open("wb") as fh:
                    shutil_data = item.file.read()
                    fh.write(shutil_data)
                _json_response(self, {"path": str(target), "url": _file_url(target)})
                return
            _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            _json_response(self, {"error": str(exc), "traceback": traceback.format_exc()}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Local3DAI web control panel.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7866)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Local3DAI web control panel: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
