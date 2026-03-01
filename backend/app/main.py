from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from loguru import logger
from PIL import Image

from .adaptive_controller import AdaptiveController, AdaptiveDecision
from .cache import FileLRUCache
from .character_db import CharacterDatabase
from .colorizer import Colorizer
from .config import get_settings
from .episodic_memory import EpisodicCharacterMemory
from .identity_enforcer import IdentityEnforcer
from .inpaint_engine import inpaint_image_with_report
from .model_manager import ModelManager
from .model_manifest import verify_manifest
from .ocr_engine import run_ocr_with_report
from .preflight import build_preflight_report
from .rating_store import RatingStore
from .scene_memory import SceneMemory
from .schemas import (
    EpisodicSummaryResponse,
    HealthResponse,
    PreflightResponse,
    ProcessReport,
    WarmupRequest,
    WarmupResponse,
)
from .advanced_render import QualityPathError, render_two_pass_variants
from .style_engine import apply_style, available_styles, get_style_package
from .structural_validator import analyze_structure
from .translate_engine import translate_regions
from .typeset_engine import typeset_image
from .upscale_engine import upscale_image_with_report
from .utils import pil_to_png_bytes, safe_image_load, sha256_bytes

settings = get_settings()
model_manager = ModelManager(settings=settings)
colorizer = Colorizer(model_manager=model_manager)
scene_memory = SceneMemory(settings=settings)
character_db = CharacterDatabase(db_path=settings.cache_dir / "character_registry.json")
identity_enforcer = IdentityEnforcer(settings=settings, character_db=character_db)
adaptive_controller = AdaptiveController(settings=settings)
episodic_memory = EpisodicCharacterMemory(
    memory_path=settings.episodic_memory_path,
    similarity_threshold=settings.episodic_similarity_threshold,
    max_palette_colors=settings.episodic_max_palette_colors,
)
cache = FileLRUCache(
    cache_dir=settings.cache_dir,
    max_items=settings.max_cache_items,
    max_size_bytes=settings.max_cache_size_gb * 1024**3,
)
rating_store = RatingStore(base_dir=settings.cache_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.log_level.upper(),
        colorize=False,
    )
    logger.info("Starting {} on {}:{}", settings.app_name, settings.host, settings.port)
    logger.info("Device={}, strict_mode={}", model_manager.device, settings.strict_pro_mode)
    yield
    model_manager.cleanup()
    logger.info("Shutdown complete")


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_image_upload(file: UploadFile) -> None:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")


def _cache_key(image_bytes: bytes, options: Dict[str, Any]) -> str:
    packed = json.dumps(options, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256_bytes(image_bytes + packed)


def _scale_regions(regions: list[dict], scale_x: float, scale_y: float) -> list[dict]:
    scaled = []
    for item in regions:
        bbox = item.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            scaled.append(item)
            continue
        x, y, w, h = bbox
        new_item = dict(item)
        new_item["bbox"] = [
            int(round(float(x) * scale_x)),
            int(round(float(y) * scale_y)),
            int(round(float(w) * scale_x)),
            int(round(float(h) * scale_y)),
        ]
        scaled.append(new_item)
    return scaled


def _parse_speaker_map(payload: Optional[str]) -> dict:
    if not payload:
        return {}
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _parse_palette_anchors(payload: Optional[str]) -> list[str]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
        if isinstance(data, list):
            anchors = [str(item).strip() for item in data if str(item).strip()]
            return anchors[:3]
    except Exception:
        pass
    return []


def _load_reference_from_path(path_value: Optional[str]) -> Optional[bytes]:
    path_value = (path_value or "").strip()
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=422, detail=f"reference path not found: {path_value}")
    return path.read_bytes()


def _normalize_shot_type(value: Optional[str]) -> str:
    raw = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "fullbody": "standing_full_body",
        "full_body": "standing_full_body",
        "standing": "standing_full_body",
        "standing_fullbody": "standing_full_body",
    }
    normalized = aliases.get(raw, raw)
    allowed = {"auto", "portrait", "half_body", "standing_full_body"}
    if normalized not in allowed:
        raise HTTPException(
            status_code=422,
            detail="shot_type must be auto|portrait|half_body|standing_full_body",
        )
    return normalized


def _looks_synthetic_pose_placeholder(image: Image.Image) -> bool:
    arr = np.array(image.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    quant = ((arr[:, :, 0] >> 4) << 8) + ((arr[:, :, 1] >> 4) << 4) + (arr[:, :, 2] >> 4)
    hist = np.bincount(quant.reshape(-1), minlength=4096).astype(np.float32)
    dominant_ratio = float(hist.max() / max(1.0, hist.sum()))
    unique_bins = int((hist > 0).sum())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    texture = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))) / 255.0
    sat_std = float(np.std(hsv[:, :, 1].astype(np.float32))) / 255.0

    return bool(
        dominant_ratio > 0.80
        and unique_bins < 80
        and edge_density < 0.06
        and texture < 0.08
        and sat_std < 0.12
    )


async def _forward_quality_to_worker(
    *,
    worker_url: str,
    auth_token: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    form_data: Dict[str, Any],
    pose_bytes: Optional[bytes] = None,
    style_ref_bytes: Optional[bytes] = None,
    character_ref_bytes: Optional[bytes] = None,
) -> Response:
    url = worker_url.rstrip("/") + "/process-image"
    files: Dict[str, tuple] = {
        "file": (filename, image_bytes, content_type),
    }
    if pose_bytes:
        files["pose_file"] = ("pose_reference.png", pose_bytes, "image/png")
    if style_ref_bytes:
        files["style_ref_file"] = ("style_reference.png", style_ref_bytes, "image/png")
    if character_ref_bytes:
        files["character_ref_file"] = ("character_reference.png", character_ref_bytes, "image/png")

    data_payload = {k: str(v) for k, v in form_data.items()}
    headers: Dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    timeout = httpx.Timeout(timeout=float(max(60, settings.request_timeout_s * 2)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, data=data_payload, files=files, headers=headers)

    if resp.status_code != 200:
        detail_payload: Any
        try:
            detail_payload = resp.json()
        except Exception:
            detail_payload = {"remote_error": resp.text}
        raise HTTPException(
            status_code=resp.status_code,
            detail={
                "error": "quality_cloud_worker_failed",
                "worker_url": worker_url,
                "remote_detail": detail_payload,
            },
        )

    out_headers = {
        "X-Cache-Hit": "0",
        "X-Remote-Worker": "1",
    }
    if "X-Process-Report" in resp.headers:
        out_headers["X-Process-Report"] = resp.headers["X-Process-Report"]
    return Response(content=resp.content, media_type="image/png", headers=out_headers)


async def _probe_quality_worker(
    *,
    worker_url: str,
    auth_token: str = "",
) -> dict:
    url = worker_url.rstrip("/") + "/health"
    headers: Dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    timeout = httpx.Timeout(timeout=8.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "error": "quality_cloud_worker_unreachable",
            "detail": str(exc),
            "worker_url": worker_url,
        }
    if resp.status_code != 200:
        return {
            "ok": False,
            "status_code": int(resp.status_code),
            "error": "quality_cloud_worker_health_failed",
            "detail": resp.text[:400],
            "worker_url": worker_url,
        }
    payload: Dict[str, Any] = {}
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = {}
    device = str(payload.get("device", "unknown"))
    if device != "cuda":
        return {
            "ok": False,
            "status_code": int(resp.status_code),
            "error": "quality_cloud_worker_not_gpu",
            "detail": {"device": device},
            "worker_url": worker_url,
        }
    return {
        "ok": True,
        "status_code": int(resp.status_code),
        "error": "",
        "detail": {"device": device},
        "worker_url": worker_url,
    }


def _quality_runtime_status() -> dict:
    worker_url = settings.quality_cloud_worker_url.strip()
    local_gpu_ready = model_manager.device == "cuda"
    worker_configured = bool(worker_url)
    if local_gpu_ready:
        mode = "local_gpu"
    elif worker_configured:
        mode = "cloud_worker"
    else:
        mode = "unavailable"
    return {
        "mode": mode,
        "local_gpu_ready": bool(local_gpu_ready),
        "quality_cloud_worker_configured": bool(worker_configured),
        "quality_cloud_worker_url": worker_url,
    }


def _run_pipeline(
    image_bytes: bytes,
    source_lang: str,
    style: str,
    colorize: bool,
    upscale: float,
    inpaint_padding: int,
    render_quality: str,
    series_id: str,
    chapter_id: str,
    page_index: int,
    enforce_identity: bool,
    speaker_map: Optional[dict],
    palette_anchors: Optional[list[str]],
    variant_count: int,
    line_weight: float,
    hatch_density: float,
    tone_intensity: float,
    screentone_strength: float,
    shot_type: str = "auto",
    pose_reference: Optional[Any] = None,
    style_reference: Optional[Any] = None,
    character_reference: Optional[Any] = None,
) -> tuple[bytes, ProcessReport]:
    timings: Dict[str, int] = {}
    engine_report: Dict[str, dict] = {}
    pipeline_start = time.perf_counter()
    image_original = safe_image_load(image_bytes)
    image = image_original.copy()
    width_orig, height_orig = image.size

    t0 = time.perf_counter()
    ocr_regions, ocr_report = run_ocr_with_report(
        image=image,
        model_manager=model_manager,
        source_lang=source_lang,
    )
    engine_report["ocr"] = ocr_report
    timings["ocr"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    lines = [str(r.get("text", "")) for r in ocr_regions]
    if settings.enable_scene_memory:
        context = scene_memory.translation_context(
            series_id=series_id,
            chapter_id=chapter_id,
            page_index=page_index,
            current_lines=lines,
        )
    else:
        context = " | ".join(line for line in lines if line.strip())
    if settings.enable_episodic_memory:
        voice_profile = episodic_memory.series_voice_profile(series_id)
        if voice_profile:
            voice_context = " | ".join([f"{k}:{v}" for k, v in voice_profile.items()])
            context = f"{context} | character_voice: {voice_context}" if context else voice_context
    translated, translation_report = translate_regions(
        ocr_regions,
        model_manager=model_manager,
        context=context,
        series_id=series_id,
        chapter_id=chapter_id,
        page_index=page_index,
        speaker_map=speaker_map or {},
        return_report=True,
    )
    translation_report["context_chars"] = len(context)
    engine_report["translation"] = translation_report
    timings["translate"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    boxes = [r["bbox"] for r in translated if len(r.get("bbox", [])) == 4]
    if boxes:
        image, _, inpaint_backend = inpaint_image_with_report(
            image=image,
            boxes=boxes,
            model_manager=model_manager,
            padding=inpaint_padding,
        )
    else:
        inpaint_backend = "skipped"
    engine_report["inpaint"] = {
        "selected": inpaint_backend,
        "regions_masked": len(boxes),
    }
    timings["inpaint"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    if upscale > 1.0:
        image, upscale_backend = upscale_image_with_report(image, scale=upscale, mode="auto")
    else:
        upscale_backend = "skipped"
    engine_report["upscale"] = {"selected": upscale_backend, "scale": float(upscale)}
    timings["upscale"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    colorize_backend = "skipped"
    if colorize:
        if colorizer.is_grayscale(image):
            image = colorizer.colorize(image, series_id=series_id)
            colorize_backend = "colorizer_diffusion"
        else:
            colorize_backend = "skipped_non_grayscale"
    engine_report["colorize"] = {
        "selected": colorize_backend,
        "enabled": bool(colorize),
    }
    timings["colorize"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    seed = int(sha256_bytes(image_bytes)[:8], 16)
    style_package_id = get_style_package(
        style,
        render_quality=render_quality,
        settings=settings,
    ).id
    if render_quality == "quality" and shot_type == "standing_full_body":
        if pose_reference is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "quality_pose_reference_required",
                    "missing": ["pose_reference"],
                    "note": "standing_full_body quality shots require a real pose reference image.",
                },
            )
        if _looks_synthetic_pose_placeholder(pose_reference):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "synthetic_pose_reference_rejected",
                    "note": "Provide a real human pose reference image, not a geometric placeholder.",
                },
            )
    if render_quality == "quality" and settings.quality_require_full_path:
        if style_reference is None and character_reference is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "quality_identity_reference_required",
                    "missing": ["style_reference_or_character_reference"],
                    "note": "Quality full-path mode requires an identity/style reference for IP-Adapter.",
                },
            )

    variant_target = max(1, int(variant_count))
    if render_quality == "preview":
        variant_target = min(variant_target, 6)
    elif render_quality == "balanced":
        variant_target = min(variant_target, 6)
    elif render_quality == "quality":
        variant_target = int(
            max(
                int(settings.quality_variant_min),
                min(int(settings.quality_variant_max), variant_target),
            )
        )
    else:
        variant_target = min(variant_target, 8)
    if model_manager.device != "cuda" and render_quality == "balanced":
        variant_target = min(variant_target, 3)

    two_pass = render_two_pass_variants(
        original=image,
        image_bytes=image_bytes,
        style_name=style,
        render_quality=render_quality,
        model_manager=model_manager,
        character_db=character_db,
        variant_count=variant_target,
        palette_anchors=list(palette_anchors or []),
        style_reference=style_reference,
        character_reference=character_reference,
        pose_reference=pose_reference,
        line_weight=float(line_weight),
        hatch_density=float(hatch_density),
        tone_intensity=float(tone_intensity),
        screentone_strength=float(screentone_strength),
        top_k=int(settings.quality_topk if render_quality == "quality" else 3),
        persist_variants=True,
    )

    image_styled = two_pass["best_image"]
    style_backend = str(two_pass.get("best_variant", {}).get("backend", "two_pass"))
    strict_render_mode = render_quality in {"final", "quality"}
    if strict_render_mode and "fallback_fast" in style_backend:
        err = "quality_fallback_backend_rejected" if render_quality == "quality" else "final_fallback_backend_rejected"
        raise QualityPathError(err)
    structural_report = analyze_structure(image, image_styled)
    preview_similarity, preview_faces, preview_matches = identity_enforcer.evaluate(
        original=image_original,
        rendered=image_styled,
        series_id=series_id,
    )

    quality_gate = dict(two_pass.get("best_variant", {}).get("quality", {}))
    if strict_render_mode and not bool(quality_gate.get("passed", False)):
        err = "quality_quality_gate_failed" if render_quality == "quality" else "final_quality_gate_failed"
        raise QualityPathError(err)
    planner_identity = {
        "best_similarity": preview_similarity,
        "faces_compared": preview_faces,
        "character_matches": preview_matches,
        "threshold": settings.identity_similarity_threshold,
        "drift_detected": preview_similarity < settings.identity_similarity_threshold and preview_faces > 0,
    }
    planner_report = {
        "enabled": True,
        "apply_rerender": not bool(quality_gate.get("passed", False)),
        "actions": [
            {
                "target": "quality_gate",
                "operation": "retry_with_lower_denoise_or_higher_control",
                "params": {
                    "variant_count": int(variant_target),
                    "best_variant": int(two_pass.get("best_variant", {}).get("index", 1)),
                },
                "reason": "Hard quality gate enforcement",
                "confidence": 0.92,
            }
        ],
        "strength_override": float(two_pass.get("best_variant", {}).get("params", {}).get("style_strength", 0.45)),
        "note": "Two-pass quality-gated planner",
        "quality_gate": quality_gate,
    }

    def rerender(attempt: int, identity_score: float):
        strength = max(0.2, float(two_pass.get("best_variant", {}).get("params", {}).get("style_strength", 0.45)) - 0.05 * attempt)
        guidance = min(12.5, float(two_pass.get("best_variant", {}).get("params", {}).get("style_guidance", 10.0)) + 0.2 * attempt)
        steps = int(min(44, float(two_pass.get("best_variant", {}).get("params", {}).get("style_steps", 32)) + 2 * attempt))
        try:
            return apply_style(
                image=image,
                style_name=style,
                model_manager=model_manager,
                render_quality=render_quality,
                seed=seed + 100 + attempt,
                strength_override=strength,
                guidance_override=guidance,
                steps_override=steps,
            )
        except RuntimeError as exc:
            if render_quality in {"final", "quality"}:
                err = "quality_identity_rerender_unavailable" if render_quality == "quality" else "final_identity_rerender_unavailable"
                raise QualityPathError(err) from exc
            raise

    identity_force_enabled = bool(enforce_identity)
    if model_manager.device != "cuda" and render_quality == "balanced":
        identity_force_enabled = False

    image_styled, identity_report = identity_enforcer.enforce(
        original=image_original,
        initial_render=image_styled,
        series_id=series_id,
        rerender_callback=rerender,
        force_enabled=identity_force_enabled,
    )

    engine_report["style"] = {
        "selected": style_backend,
        "style_id": style_package_id,
        "render_quality": render_quality,
    }
    engine_report["request"] = {
        "shot_type": shot_type,
        "identity_enforce_runtime": bool(identity_force_enabled),
        "style_controls": {
            "line_weight": float(line_weight),
            "hatch_density": float(hatch_density),
            "tone_intensity": float(tone_intensity),
            "screentone_strength": float(screentone_strength),
        },
    }
    engine_report["adaptive_controller"] = {
        "enabled": True,
        "decision": {
            "seed": int(two_pass.get("best_variant", {}).get("seed", seed)),
            "variant_index": int(two_pass.get("best_variant", {}).get("index", 1)),
            "params": dict(two_pass.get("best_variant", {}).get("params", {})),
        },
    }
    engine_report["planner"] = planner_report
    engine_report["two_pass"] = {
        "run_dir": two_pass.get("run_dir", ""),
        "metadata": two_pass.get("metadata", {}),
        "top_variants": two_pass.get("top_variants", []),
    }
    image = image_styled
    timings["style"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    if translated:
        sx = image.width / float(width_orig)
        sy = image.height / float(height_orig)
        translated_scaled = _scale_regions(translated, sx, sy)
        image = typeset_image(image, translated_scaled, fonts_dir=settings.fonts_dir)
        typeset_backend = "pillow"
    else:
        typeset_backend = "skipped"
    engine_report["typeset"] = {
        "selected": typeset_backend,
        "regions": len(translated),
    }
    timings["typeset"] = int((time.perf_counter() - t0) * 1000)
    episodic_report = {
        "enabled": bool(settings.enable_episodic_memory),
        "series_id": series_id,
        "faces_detected": 0,
        "characters_total": 0,
        "new_characters": 0,
        "matches": 0,
    }
    episodic_palette: list[str] = []
    if settings.enable_episodic_memory:
        episodic_update = episodic_memory.update_from_panel(
            series_id=series_id,
            image=image_original,
            character_db=character_db,
            speaker_notes={str(k): str(v) for k, v in (speaker_map or {}).items()},
            page_index=page_index,
        )
        episodic_report.update(episodic_update)
        episodic_palette = episodic_memory.series_palette(
            series_id=series_id,
            max_colors=settings.episodic_max_palette_colors,
        )
    engine_report["episodic_memory"] = episodic_report

    if settings.enable_scene_memory:
        scene_palette = scene_memory.palette_anchor(series_id, chapter_id, page_index)
        merged_palette = []
        for color in list(episodic_palette) + list(scene_palette):
            if color not in merged_palette:
                merged_palette.append(color)
        scene_memory.update_page(
            series_id=series_id,
            chapter_id=chapter_id,
            page_index=page_index,
            style_id=style,
            source_lines=[str(r.get("text", "")).strip() for r in ocr_regions],
            translated_lines=[str(r.get("translated_text", "")).strip() for r in translated],
            palette_anchor=merged_palette,
            speaker_notes={str(k): str(v) for k, v in (speaker_map or {}).items()},
        )

    out_bytes = pil_to_png_bytes(image)
    timings["total"] = int((time.perf_counter() - pipeline_start) * 1000)
    if settings.enable_adaptive_controller:
        adaptive_decision = engine_report.get("adaptive_controller", {}).get("decision")
        if adaptive_decision is not None:
            if isinstance(adaptive_decision, dict):
                params = dict(adaptive_decision.get("params", {}))
                decision_obj = AdaptiveDecision(
                    strength=float(params.get("style_strength", 0.45)),
                    guidance_scale=float(params.get("style_guidance", 9.0)),
                    steps=int(params.get("style_steps", 24)),
                    confidence=0.72,
                    reason="two_pass_best_variant",
                )
            else:
                decision_obj = adaptive_decision
            adaptive_controller.record_outcome(
                style_id=style_package_id,
                decision=decision_obj,
                structural_report=structural_report.to_dict(),
                identity_report=identity_report.to_dict(),
                style_backend=style_backend,
            )

    report = ProcessReport(
        cached=False,
        image_width=image.width,
        image_height=image.height,
        ocr_regions=len(ocr_regions),
        translated_regions=len([r for r in translated if r.get("translated_text")]),
        style_id=style,
        render_quality=render_quality,
        missing_capabilities=[],
        identity_report=identity_report.to_dict(),
        structural_report=structural_report.to_dict(),
        planner_report=planner_report,
        episodic_report=episodic_report,
        engine_report=engine_report,
        timings_ms=timings,
    )
    return out_bytes, report


def _strict_blockers(
    render_quality: str,
    source_lang: str,
    style: str,
    colorize: bool,
) -> list[str]:
    need_diffusion = bool(render_quality in {"balanced", "final", "quality"} and style != "original")
    blockers = model_manager.strict_mode_blockers(
        render_quality=render_quality,
        need_translation=True,
        need_ocr=source_lang in {"auto", "ko", "ja"},
        need_diffusion=need_diffusion or colorize,
    )
    if render_quality == "quality":
        quality_missing = model_manager.missing_requirements(
            render_quality=render_quality,
            need_translation=True,
            need_ocr=source_lang in {"auto", "ko", "ja"},
            need_diffusion=True,
        )
        blockers = sorted(set(blockers + quality_missing))
    if settings.strict_pro_mode and settings.local_models_only:
        manifest = verify_manifest(
            models_dir=settings.models_dir,
            manifest_path=settings.model_manifest_path,
            include_file_hashes=False,
            hash_max_mb=64,
        )
        if not manifest.get("manifest_exists", False):
            blockers.append("model_manifest_not_found")
        elif not manifest.get("matches", False):
            blockers.append("model_manifest_mismatch")
    return sorted(set(blockers))


def _normalize_warmup_payload(payload: WarmupRequest) -> WarmupRequest:
    quality = (payload.render_quality or "balanced").strip().lower()
    style = (payload.style or "original").strip().lower()
    source = (payload.source_lang or "auto").strip().lower()
    if quality not in {"preview", "balanced", "final", "quality"}:
        raise HTTPException(status_code=422, detail="render_quality must be preview|balanced|final|quality")
    if source not in {"auto", "ja", "ko"}:
        raise HTTPException(status_code=422, detail="source_lang must be auto|ja|ko")
    return WarmupRequest(
        render_quality=quality,
        style=style,
        source_lang=source,
        colorize=bool(payload.colorize),
        include_llm=bool(payload.include_llm),
        strict=bool(payload.strict),
    )


def _build_warmup_response(payload: WarmupRequest) -> WarmupResponse:
    source_lang = payload.source_lang
    need_ocr = source_lang in {"auto", "ja", "ko"}
    style = payload.style
    quality = payload.render_quality
    need_diffusion = bool(payload.colorize or style != "original" or quality in {"balanced", "final", "quality"})
    need_inpaint = bool(need_ocr)

    warmup_result = model_manager.warmup(
        include_ocr=need_ocr,
        include_translation=True,
        include_diffusion=need_diffusion,
        include_inpaint=need_inpaint,
        include_llm=payload.include_llm,
        require_ocr=need_ocr,
        require_translation=True,
        require_diffusion=need_diffusion,
        require_inpaint=need_inpaint and need_diffusion,
    )

    blockers = model_manager.missing_requirements(
        render_quality=quality,
        need_translation=True,
        need_ocr=need_ocr,
        need_diffusion=need_diffusion,
    )
    if payload.strict:
        if settings.local_models_only:
            manifest = verify_manifest(
                models_dir=settings.models_dir,
                manifest_path=settings.model_manifest_path,
                include_file_hashes=False,
                hash_max_mb=64,
            )
            if not manifest.get("manifest_exists", False):
                blockers = sorted(set(blockers + ["model_manifest_not_found"]))
            elif not manifest.get("matches", False):
                blockers = sorted(set(blockers + ["model_manifest_mismatch"]))
    elif not payload.strict:
        blockers = _strict_blockers(
            render_quality=quality,
            source_lang=source_lang,
            style=style,
            colorize=payload.colorize,
        )

    missing = list(warmup_result.get("required_missing", []))
    is_ok = bool(warmup_result.get("ok", False) and not blockers and not missing)
    status = "ready" if is_ok else ("blocked" if blockers else "degraded")

    return WarmupResponse(
        status=status,
        ok=is_ok,
        device=str(warmup_result.get("device", model_manager.device)),
        strict_mode=bool(settings.strict_pro_mode or payload.strict),
        requested={
            "render_quality": quality,
            "style": style,
            "source_lang": source_lang,
            "colorize": bool(payload.colorize),
            "include_llm": bool(payload.include_llm),
            "strict": bool(payload.strict),
            "need_ocr": need_ocr,
            "need_diffusion": need_diffusion,
        },
        blockers=blockers,
        missing_required=missing,
        steps=list(warmup_result.get("steps", [])),
        timings_ms={"total": int(warmup_result.get("total_ms", 0))},
        capabilities=dict(warmup_result.get("capabilities", {})),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    t0 = time.perf_counter()
    return HealthResponse(
        status="ok",
        time_ms=int((time.perf_counter() - t0) * 1000),
        device=model_manager.device,
        strict_mode=settings.strict_pro_mode,
    )


@app.get("/styles")
async def styles() -> dict:
    return {"styles": available_styles()}


@app.get("/styles/full")
async def styles_full() -> dict:
    packages = []
    for style_id in available_styles():
        package = get_style_package(style_id, render_quality="balanced", settings=settings)
        packages.append(package.to_dict())
    return {"styles": packages}


@app.get("/capabilities")
async def capabilities() -> dict:
    return model_manager.capabilities(refresh=True)


@app.get("/preflight", response_model=PreflightResponse)
async def preflight() -> PreflightResponse:
    return PreflightResponse(**build_preflight_report(settings=settings, model_manager=model_manager))


@app.get("/quality/readiness")
async def quality_readiness() -> dict:
    status = _quality_runtime_status()
    if status["mode"] == "cloud_worker":
        probe = await _probe_quality_worker(
            worker_url=settings.quality_cloud_worker_url.strip(),
            auth_token=settings.quality_cloud_worker_token.strip(),
        )
        status["worker_probe"] = probe
    else:
        status["worker_probe"] = {"ok": False, "error": "not_applicable", "detail": ""}
    status["quality_ready"] = bool(
        status["mode"] == "local_gpu" or bool(status.get("worker_probe", {}).get("ok", False))
    )
    return status


@app.get("/warmup", response_model=WarmupResponse)
async def warmup_get(
    render_quality: str = "balanced",
    style: str = "original",
    source_lang: str = "auto",
    colorize: bool = False,
    include_llm: bool = False,
    strict: bool = False,
) -> WarmupResponse:
    payload = _normalize_warmup_payload(
        WarmupRequest(
            render_quality=render_quality,
            style=style,
            source_lang=source_lang,
            colorize=colorize,
            include_llm=include_llm,
            strict=strict,
        )
    )
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_build_warmup_response, payload),
            timeout=settings.warmup_timeout_s,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Warmup timed out")


@app.post("/warmup", response_model=WarmupResponse)
async def warmup_post(payload: WarmupRequest) -> WarmupResponse:
    normalized = _normalize_warmup_payload(payload)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_build_warmup_response, normalized),
            timeout=settings.warmup_timeout_s,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Warmup timed out")


@app.get("/memory/{series_id}", response_model=EpisodicSummaryResponse)
async def memory_summary(series_id: str) -> EpisodicSummaryResponse:
    return EpisodicSummaryResponse(**episodic_memory.summary(series_id))


@app.get("/memory")
async def memory_all() -> dict:
    return {"series": episodic_memory.all_series()}


@app.get("/controller")
async def controller_summary() -> dict:
    return adaptive_controller.summary()


@app.get("/controller/{style_id}")
async def controller_style(style_id: str) -> dict:
    resolved = get_style_package(style_id, render_quality="balanced", settings=settings).id
    return adaptive_controller.summary_style(resolved)


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    pose_file: Optional[UploadFile] = File(None),
    style_ref_file: Optional[UploadFile] = File(None),
    character_ref_file: Optional[UploadFile] = File(None),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en"),
    style: str = Form("original"),
    colorize: bool = Form(False),
    upscale: float = Form(1.0),
    inpaint_padding: int = Form(8),
    render_quality: str = Form("balanced"),
    series_id: str = Form("default_series"),
    chapter_id: str = Form("default_chapter"),
    page_index: int = Form(0),
    enforce_identity: bool = Form(True),
    speaker_map_json: Optional[str] = Form(None),
    palette_anchors_json: Optional[str] = Form(None),
    pose_ref_path: Optional[str] = Form(None),
    style_ref_path: Optional[str] = Form(None),
    character_ref_path: Optional[str] = Form(None),
    line_weight: float = Form(1.0),
    hatch_density: float = Form(0.52),
    tone_intensity: float = Form(0.42),
    screentone_strength: float = Form(0.16),
    shot_type: str = Form("auto"),
    variant_count: int = Form(1),
    x_preferred_style: Optional[str] = Header(default=None),
) -> Response:
    _validate_image_upload(file)
    if target_lang != "en":
        raise HTTPException(status_code=400, detail="Only target_lang=en is currently supported")
    if not (1.0 <= upscale <= 4.0):
        raise HTTPException(status_code=422, detail="upscale must be between 1.0 and 4.0")
    if render_quality not in {"preview", "balanced", "final", "quality"}:
        raise HTTPException(status_code=422, detail="render_quality must be preview|balanced|final|quality")
    if not (0.5 <= float(line_weight) <= 2.0):
        raise HTTPException(status_code=422, detail="line_weight must be between 0.5 and 2.0")
    if not (0.1 <= float(hatch_density) <= 1.0):
        raise HTTPException(status_code=422, detail="hatch_density must be between 0.1 and 1.0")
    if not (0.0 <= float(tone_intensity) <= 1.0):
        raise HTTPException(status_code=422, detail="tone_intensity must be between 0.0 and 1.0")
    if not (0.0 <= float(screentone_strength) <= 1.0):
        raise HTTPException(status_code=422, detail="screentone_strength must be between 0.0 and 1.0")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    normalized_shot_type = _normalize_shot_type(shot_type)
    line_weight = float(np.clip(line_weight, 0.5, 2.0))
    hatch_density = float(np.clip(hatch_density, 0.1, 1.0))
    tone_intensity = float(np.clip(tone_intensity, 0.0, 1.0))
    screentone_strength = float(np.clip(screentone_strength, 0.0, 1.0))

    pose_bytes = await pose_file.read() if pose_file is not None else None
    style_ref_bytes = await style_ref_file.read() if style_ref_file is not None else None
    character_ref_bytes = await character_ref_file.read() if character_ref_file is not None else None
    if not pose_bytes:
        pose_bytes = _load_reference_from_path(pose_ref_path)
    if not style_ref_bytes:
        style_ref_bytes = _load_reference_from_path(style_ref_path)
    if not character_ref_bytes:
        character_ref_bytes = _load_reference_from_path(character_ref_path)

    pose_reference = safe_image_load(pose_bytes) if pose_bytes else None
    style_reference = safe_image_load(style_ref_bytes) if style_ref_bytes else None
    character_reference = safe_image_load(character_ref_bytes) if character_ref_bytes else None

    selected_style = (x_preferred_style or style or "original").strip().lower()
    quality_mode = render_quality == "quality"
    strict_gpu_mode = render_quality in {"final", "quality"}
    quality_worker_url = settings.quality_cloud_worker_url.strip()
    quality_worker_token = settings.quality_cloud_worker_token.strip()

    if quality_mode and normalized_shot_type == "standing_full_body":
        if pose_reference is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "quality_pose_reference_required",
                    "missing": ["pose_reference"],
                    "note": "standing_full_body quality shots require a real pose reference image.",
                },
            )
        if _looks_synthetic_pose_placeholder(pose_reference):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "synthetic_pose_reference_rejected",
                    "note": "Provide a real human pose reference image, not a geometric placeholder.",
                },
            )
    if quality_mode and settings.quality_require_full_path:
        if style_reference is None and character_reference is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "quality_identity_reference_required",
                    "missing": ["style_reference_or_character_reference"],
                    "note": "Quality full-path mode requires an identity/style reference for IP-Adapter.",
                },
            )

    if strict_gpu_mode and model_manager.device != "cuda":
        if quality_worker_url:
            probe = await _probe_quality_worker(
                worker_url=quality_worker_url,
                auth_token=quality_worker_token,
            )
            if not bool(probe.get("ok", False)):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": str(probe.get("error", "quality_cloud_worker_unreachable")),
                        "worker_url": quality_worker_url,
                        "probe": probe,
                        "note": "Final/quality render mode requires a CUDA-capable worker when local GPU is unavailable.",
                    },
                )
            forward_form = {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "style": selected_style,
                "colorize": str(bool(colorize)).lower(),
                "upscale": str(upscale),
                "inpaint_padding": str(inpaint_padding),
                "render_quality": render_quality,
                "series_id": series_id,
                "chapter_id": chapter_id,
                "page_index": str(page_index),
                "enforce_identity": str(bool(enforce_identity)).lower(),
                "speaker_map_json": speaker_map_json or "",
                "palette_anchors_json": palette_anchors_json or "",
                "line_weight": str(line_weight),
                "hatch_density": str(hatch_density),
                "tone_intensity": str(tone_intensity),
                "screentone_strength": str(screentone_strength),
                "shot_type": normalized_shot_type,
                "variant_count": str(variant_count),
            }
            return await _forward_quality_to_worker(
                worker_url=quality_worker_url,
                auth_token=quality_worker_token,
                image_bytes=data,
                filename=file.filename or "panel.png",
                content_type=file.content_type or "image/png",
                form_data=forward_form,
                pose_bytes=pose_bytes,
                style_ref_bytes=style_ref_bytes,
                character_ref_bytes=character_ref_bytes,
            )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "strict_render_mode_requires_gpu",
                "missing": ["quality_requires_cuda_gpu" if quality_mode else "final_requires_cuda_gpu"],
                "note": "Final/quality render modes are strict and do not fall back to degraded CPU generation.",
                "actions": [
                    "Run on a CUDA GPU machine",
                    "Configure QUALITY_CLOUD_WORKER_URL to a GPU backend",
                ],
            },
        )

    blockers = _strict_blockers(
        render_quality=render_quality,
        source_lang=source_lang,
        style=selected_style,
        colorize=colorize,
    )
    if blockers:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Strict production mode requirements not met",
                "missing": blockers,
                "capabilities": model_manager.capabilities(refresh=True),
            },
        )

    palette_anchors = _parse_palette_anchors(palette_anchors_json)
    variant_cap = int(settings.quality_variant_max if quality_mode else 12)
    variant_count = int(max(1, min(variant_cap, int(variant_count))))

    options = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "style": selected_style,
        "colorize": colorize,
        "upscale": upscale,
        "inpaint_padding": inpaint_padding,
        "render_quality": render_quality,
        "series_id": series_id,
        "chapter_id": chapter_id,
        "page_index": page_index,
        "enforce_identity": enforce_identity,
        "speaker_map_json": speaker_map_json or "",
        "palette_anchors_json": palette_anchors_json or "",
        "line_weight": line_weight,
        "hatch_density": hatch_density,
        "tone_intensity": tone_intensity,
        "screentone_strength": screentone_strength,
        "shot_type": normalized_shot_type,
        "variant_count": variant_count,
        "pose_ref_hash": sha256_bytes(pose_bytes)[:16] if pose_bytes else "",
        "style_ref_hash": sha256_bytes(style_ref_bytes)[:16] if style_ref_bytes else "",
        "character_ref_hash": sha256_bytes(character_ref_bytes)[:16] if character_ref_bytes else "",
    }
    key = _cache_key(data, options)
    cached = cache.get(key)
    if cached:
        return Response(
            content=cached,
            media_type="image/png",
            headers={"X-Cache-Hit": "1"},
        )

    speaker_map = _parse_speaker_map(speaker_map_json)
    try:
        out_bytes, report = await asyncio.wait_for(
            asyncio.to_thread(
                _run_pipeline,
                data,
                source_lang,
                selected_style,
                colorize,
                upscale,
                inpaint_padding,
                render_quality,
                series_id,
                chapter_id,
                page_index,
                enforce_identity,
                speaker_map,
                palette_anchors,
                variant_count,
                line_weight,
                hatch_density,
                tone_intensity,
                screentone_strength,
                normalized_shot_type,
                pose_reference,
                style_reference,
                character_reference,
            ),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Image processing timed out")
    except HTTPException:
        raise
    except QualityPathError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "strict_render_path_unavailable",
                "missing": [str(exc)],
                "note": "Final/quality render modes are strict and do not fall back to degraded generation.",
            },
        )
    except Exception as exc:
        logger.exception("Pipeline failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")

    cache.put(key, out_bytes)
    headers = {"X-Cache-Hit": "0", "X-Process-Report": report.model_dump_json()}
    return Response(content=out_bytes, media_type="image/png", headers=headers)


@app.exception_handler(Exception)
async def generic_exception_handler(_, exc: Exception):
    logger.exception("Unhandled server exception: {}", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/review", response_class=HTMLResponse)
async def review_home() -> HTMLResponse:
    html = """
    <html><head><title>MangaShift A/B Review</title>
    <style>
    body{font-family:Segoe UI,Arial,sans-serif;background:#0f1014;color:#e7e9ef;padding:18px}
    input,button,select{padding:8px;margin:4px;background:#161824;color:#e7e9ef;border:1px solid #33384f}
    .row{display:flex;gap:12px;align-items:flex-start}
    .card{background:#151925;border:1px solid #2d3347;padding:10px;border-radius:8px}
    img{max-width:47vw;border:1px solid #2c3242}
    pre{white-space:pre-wrap}
    </style></head>
    <body>
      <h2>MangaShift Variant Review</h2>
      <p>Load a run id from <code>/review/runs</code>, compare two variants, and submit winner.</p>
      <div class='card'>
        <label>Run ID</label><input id='run' size='24'/>
        <button onclick='loadRun()'>Load</button>
        <span id='status'></span>
      </div>
      <div class='card'>
        <label>Left idx</label><input id='left' value='1' size='4'/>
        <label>Right idx</label><input id='right' value='2' size='4'/>
        <label>Winner</label>
        <select id='winner'><option value='left'>left</option><option value='right'>right</option></select>
        <button onclick='rate()'>Submit Rating</button>
      </div>
      <div class='row'>
        <div class='card'><div>Left</div><img id='imgL'/></div>
        <div class='card'><div>Right</div><img id='imgR'/></div>
      </div>
      <pre id='meta'></pre>
    <script>
    let meta={};
    async function loadRun(){
      const id=document.getElementById('run').value.trim();
      if(!id){return;}
      const r=await fetch('/review/run/'+id);
      meta=await r.json();
      document.getElementById('meta').textContent=JSON.stringify(meta,null,2);
      showImages();
      document.getElementById('status').textContent=' loaded';
    }
    function showImages(){
      const id=document.getElementById('run').value.trim();
      const l=parseInt(document.getElementById('left').value||'1',10);
      const rr=parseInt(document.getElementById('right').value||'2',10);
      document.getElementById('imgL').src='/review/image/'+id+'/variant_'+String(l).padStart(2,'0')+'_final.png';
      document.getElementById('imgR').src='/review/image/'+id+'/variant_'+String(rr).padStart(2,'0')+'_final.png';
    }
    document.getElementById('left').addEventListener('change',showImages);
    document.getElementById('right').addEventListener('change',showImages);
    async function rate(){
      const id=document.getElementById('run').value.trim();
      const payload={
        run_id:id,
        left_index:parseInt(document.getElementById('left').value||'1',10),
        right_index:parseInt(document.getElementById('right').value||'2',10),
        winner:document.getElementById('winner').value,
      };
      await fetch('/review/rate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      alert('saved');
    }
    </script></body></html>
    """
    return HTMLResponse(content=html)


@app.get("/review/runs")
async def review_runs() -> dict:
    return {"runs": rating_store.list_runs(), "ratings_count": len(rating_store.recent(limit=10000))}


@app.get("/review/run/{run_id}")
async def review_run(run_id: str) -> dict:
    payload = rating_store.run_metadata(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="run not found")
    return payload


@app.get("/review/image/{run_id}/{name}")
async def review_image(run_id: str, name: str) -> Response:
    run_dir = settings.cache_dir / "variant_runs" / str(run_id)
    file_path = run_dir / name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return Response(content=file_path.read_bytes(), media_type="image/png")


@app.post("/review/rate")
async def review_rate(payload: dict = Body(default_factory=dict)) -> dict:
    run_id = str(payload.get("run_id", "")).strip()
    if not run_id:
        raise HTTPException(status_code=422, detail="run_id required")
    left_index = int(payload.get("left_index", 1))
    right_index = int(payload.get("right_index", 2))
    winner = str(payload.get("winner", "left")).strip().lower()
    if winner not in {"left", "right"}:
        raise HTTPException(status_code=422, detail="winner must be left|right")
    notes = str(payload.get("notes", "")).strip()

    rating_store.append(
        {
            "run_id": run_id,
            "left_index": left_index,
            "right_index": right_index,
            "winner": winner,
            "notes": notes,
        }
    )
    return {"ok": True}
