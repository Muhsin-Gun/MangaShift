from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from fastapi.responses import Response
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402


def build_sample_image_bytes() -> bytes:
    img = Image.new("RGB", (560, 820), "white")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((40, 50, 520, 250), radius=30, outline="black", width=3, fill="white")
    draw.rounded_rectangle((90, 330, 500, 540), radius=30, outline="black", width=3, fill="white")
    draw.text((120, 120), "annyeong!", fill="black")
    draw.text((140, 390), "konnichiwa", fill="black")
    draw.rectangle((15, 650, 540, 780), fill=(235, 235, 235), outline="black", width=3)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "strict_mode" in body


def test_capabilities_endpoint():
    r = client.get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert "modules" in body
    assert "device" in body


def test_controller_endpoint_shape():
    r = client.get("/controller")
    assert r.status_code == 200
    body = r.json()
    assert "styles" in body
    assert "style_count" in body


def test_preflight_endpoint_shape():
    r = client.get("/preflight")
    assert r.status_code == 200
    body = r.json()
    assert "tier" in body
    assert "readiness_score" in body
    assert "blockers" in body
    assert "warnings" in body
    assert "actions" in body
    assert "model_paths" in body
    assert "model_manifest" in body
    assert "feature_flags" in body


def test_warmup_get_shape():
    r = client.get("/warmup")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "ok" in body
    assert "requested" in body
    assert "steps" in body
    assert "timings_ms" in body
    assert "capabilities" in body


def test_warmup_post_shape():
    payload = {
        "render_quality": "preview",
        "style": "original",
        "source_lang": "auto",
        "colorize": False,
        "strict": False,
    }
    r = client.post("/warmup", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["requested"]["render_quality"] == "preview"
    assert "missing_required" in body
    assert "blockers" in body


def test_styles():
    r = client.get("/styles")
    assert r.status_code == 200
    styles = r.json()["styles"]
    assert "smooth" in styles
    assert "realistic" in styles
    assert len(styles) >= 18


def test_styles_full_shape():
    r = client.get("/styles/full")
    assert r.status_code == 200
    styles = r.json()["styles"]
    assert isinstance(styles, list)
    assert len(styles) >= 18
    first = styles[0]
    assert "id" in first
    assert "prompt_template" in first
    assert "img2img_params" in first


def test_process_image_png_and_changed():
    image_bytes = build_sample_image_bytes()
    files = {"file": ("sample.png", image_bytes, "image/png")}
    data = {
        "source_lang": "auto",
        "target_lang": "en",
        "style": "gritty",
        "colorize": "false",
        "upscale": "1.0",
        "line_weight": "1.15",
        "hatch_density": "0.6",
        "tone_intensity": "0.5",
        "screentone_strength": "0.2",
    }
    r = client.post("/process-image", files=files, data=data)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")
    if r.headers.get("X-Cache-Hit") == "0":
        assert "X-Process-Report" in r.headers
        report = json.loads(r.headers["X-Process-Report"])
        assert "structural_report" in report
        assert "engine_report" in report
        assert "planner_report" in report
        assert "episodic_report" in report
        assert "planner" in report["engine_report"]
        assert "episodic_memory" in report["engine_report"]
        assert "adaptive_controller" in report["engine_report"]
        req = report["engine_report"].get("request", {})
        assert "style_controls" in req
    out = Image.open(io.BytesIO(r.content)).convert("RGB")
    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    assert out.size == src.size
    diff = np.abs(np.array(out, dtype=np.int16) - np.array(src, dtype=np.int16)).mean()
    assert diff > 0.5


def test_process_cache_hit():
    image_bytes = build_sample_image_bytes()
    files = {"file": ("sample.png", image_bytes, "image/png")}
    data = {
        "source_lang": "auto",
        "target_lang": "en",
        "style": "smooth",
        "colorize": "false",
        "upscale": "1.0",
        "line_weight": "1.1",
        "hatch_density": "0.55",
        "tone_intensity": "0.45",
        "screentone_strength": "0.15",
    }
    r1 = client.post("/process-image", files=files, data=data)
    assert r1.status_code == 200
    r2 = client.post("/process-image", files=files, data=data)
    assert r2.status_code == 200
    assert r2.headers.get("X-Cache-Hit") == "1"


def test_strict_mode_blocks_when_requirements_missing():
    from app.main import settings

    prev_strict = settings.strict_pro_mode
    prev_req_diffusion = settings.strict_require_diffusion
    try:
        settings.strict_pro_mode = True
        settings.strict_require_diffusion = True
        image_bytes = build_sample_image_bytes()
        files = {"file": ("sample.png", image_bytes, "image/png")}
        data = {
            "source_lang": "auto",
            "target_lang": "en",
            "style": "cinematic",
            "colorize": "false",
            "upscale": "1.0",
            "render_quality": "final",
        }
        r = client.post("/process-image", files=files, data=data)
        assert r.status_code in (200, 503)
        if r.status_code == 503:
            payload = r.json()["detail"]
            assert "missing" in payload
    finally:
        settings.strict_pro_mode = prev_strict
        settings.strict_require_diffusion = prev_req_diffusion


def test_memory_endpoint_shape():
    r = client.get("/memory/default_series")
    assert r.status_code == 200
    body = r.json()
    assert "series_id" in body
    assert "characters_total" in body
    assert "seen_faces_total" in body


def test_quality_readiness_endpoint_shape():
    r = client.get("/quality/readiness")
    assert r.status_code == 200
    body = r.json()
    assert "mode" in body
    assert "quality_ready" in body
    assert "local_gpu_ready" in body
    assert "worker_probe" in body


def test_quality_mode_forwards_to_cloud_worker_when_probe_ok(monkeypatch):
    from app.main import model_manager, settings

    prev_url = settings.quality_cloud_worker_url
    prev_token = settings.quality_cloud_worker_token
    prev_device = model_manager.device
    try:
        settings.quality_cloud_worker_url = "http://fake-worker.local:9000"
        settings.quality_cloud_worker_token = ""
        model_manager.device = "cpu"

        async def _fake_probe(*, worker_url: str, auth_token: str = "") -> dict:
            return {
                "ok": True,
                "status_code": 200,
                "error": "",
                "detail": {"device": "cuda"},
                "worker_url": worker_url,
            }

        async def _fake_forward(**kwargs) -> Response:
            return Response(
                content=build_sample_image_bytes(),
                media_type="image/png",
                headers={"X-Remote-Worker": "1"},
            )

        monkeypatch.setattr("app.main._probe_quality_worker", _fake_probe)
        monkeypatch.setattr("app.main._forward_quality_to_worker", _fake_forward)

        image_bytes = build_sample_image_bytes()
        files = {
            "file": ("sample.png", image_bytes, "image/png"),
            "style_ref_file": ("style.png", image_bytes, "image/png"),
        }
        data = {
            "source_lang": "auto",
            "target_lang": "en",
            "style": "realistic",
            "render_quality": "quality",
            "shot_type": "auto",
            "variant_count": "16",
            "upscale": "1.0",
            "colorize": "false",
        }
        r = client.post("/process-image", files=files, data=data)
        assert r.status_code == 200, r.text
        assert r.headers.get("X-Remote-Worker") == "1"
    finally:
        settings.quality_cloud_worker_url = prev_url
        settings.quality_cloud_worker_token = prev_token
        model_manager.device = prev_device


def test_final_mode_rejects_cpu_without_gpu_worker():
    from app.main import model_manager, settings

    prev_url = settings.quality_cloud_worker_url
    prev_token = settings.quality_cloud_worker_token
    prev_device = model_manager.device
    try:
        settings.quality_cloud_worker_url = ""
        settings.quality_cloud_worker_token = ""
        model_manager.device = "cpu"

        image_bytes = build_sample_image_bytes()
        files = {"file": ("sample.png", image_bytes, "image/png")}
        data = {
            "source_lang": "auto",
            "target_lang": "en",
            "style": "realistic",
            "render_quality": "final",
            "shot_type": "auto",
            "variant_count": "8",
            "upscale": "1.0",
            "colorize": "false",
        }
        r = client.post("/process-image", files=files, data=data)
        assert r.status_code == 503
        body = r.json()
        assert body["detail"]["error"] == "strict_render_mode_requires_gpu"
    finally:
        settings.quality_cloud_worker_url = prev_url
        settings.quality_cloud_worker_token = prev_token
        model_manager.device = prev_device


def test_final_mode_forwards_to_cloud_worker_when_probe_ok(monkeypatch):
    from app.main import model_manager, settings

    prev_url = settings.quality_cloud_worker_url
    prev_token = settings.quality_cloud_worker_token
    prev_device = model_manager.device
    try:
        settings.quality_cloud_worker_url = "http://fake-worker.local:9000"
        settings.quality_cloud_worker_token = ""
        model_manager.device = "cpu"

        async def _fake_probe(*, worker_url: str, auth_token: str = "") -> dict:
            return {
                "ok": True,
                "status_code": 200,
                "error": "",
                "detail": {"device": "cuda"},
                "worker_url": worker_url,
            }

        async def _fake_forward(**kwargs) -> Response:
            return Response(
                content=build_sample_image_bytes(),
                media_type="image/png",
                headers={"X-Remote-Worker": "1"},
            )

        monkeypatch.setattr("app.main._probe_quality_worker", _fake_probe)
        monkeypatch.setattr("app.main._forward_quality_to_worker", _fake_forward)

        image_bytes = build_sample_image_bytes()
        files = {"file": ("sample.png", image_bytes, "image/png")}
        data = {
            "source_lang": "auto",
            "target_lang": "en",
            "style": "realistic",
            "render_quality": "final",
            "shot_type": "auto",
            "variant_count": "8",
            "upscale": "1.0",
            "colorize": "false",
        }
        r = client.post("/process-image", files=files, data=data)
        assert r.status_code == 200, r.text
        assert r.headers.get("X-Remote-Worker") == "1"
    finally:
        settings.quality_cloud_worker_url = prev_url
        settings.quality_cloud_worker_token = prev_token
        model_manager.device = prev_device


def test_quality_mode_rejects_unreachable_cloud_worker(monkeypatch):
    from app.main import model_manager, settings

    prev_url = settings.quality_cloud_worker_url
    prev_token = settings.quality_cloud_worker_token
    prev_device = model_manager.device
    try:
        settings.quality_cloud_worker_url = "http://fake-worker.local:9000"
        settings.quality_cloud_worker_token = ""
        model_manager.device = "cpu"

        async def _fake_probe(*, worker_url: str, auth_token: str = "") -> dict:
            return {
                "ok": False,
                "status_code": 0,
                "error": "quality_cloud_worker_unreachable",
                "detail": "connection refused",
                "worker_url": worker_url,
            }

        monkeypatch.setattr("app.main._probe_quality_worker", _fake_probe)

        image_bytes = build_sample_image_bytes()
        files = {
            "file": ("sample.png", image_bytes, "image/png"),
            "style_ref_file": ("style.png", image_bytes, "image/png"),
        }
        data = {
            "source_lang": "auto",
            "target_lang": "en",
            "style": "realistic",
            "render_quality": "quality",
            "shot_type": "auto",
            "variant_count": "16",
            "upscale": "1.0",
            "colorize": "false",
        }
        r = client.post("/process-image", files=files, data=data)
        assert r.status_code == 503
        body = r.json()
        assert body["detail"]["error"] == "quality_cloud_worker_unreachable"
    finally:
        settings.quality_cloud_worker_url = prev_url
        settings.quality_cloud_worker_token = prev_token
        model_manager.device = prev_device
