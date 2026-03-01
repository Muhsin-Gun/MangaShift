from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageEnhance, ImageFilter

from .config import Settings
from .model_manager import ModelManager
from .style_packages import StylePackage, StylePackageManager
from .utils import pil_to_png_bytes, safe_image_load


@lru_cache(maxsize=1)
def _style_manager() -> StylePackageManager:
    from .config import get_settings

    return StylePackageManager(get_settings())


def available_styles() -> list[str]:
    return _style_manager().list_ids()


def get_style_package(style_name: str, render_quality: str, settings: Settings) -> StylePackage:
    manager = _style_manager()
    # If settings changed in-memory, re-apply quality with supplied settings.
    package = manager.get(style_name, quality=render_quality)
    return package.with_quality(render_quality, settings=settings)


def _apply_post_filters(image: Image.Image, package: StylePackage, seed: Optional[int]) -> Image.Image:
    out = image.convert("RGB")
    filters = package.post_filters or {}
    if "contrast" in filters:
        out = ImageEnhance.Contrast(out).enhance(1.0 + float(filters["contrast"]))
    if "saturation" in filters:
        out = ImageEnhance.Color(out).enhance(max(0.0, 1.0 + float(filters["saturation"])))
    if "sharpness" in filters:
        out = ImageEnhance.Sharpness(out).enhance(max(0.0, 1.0 + float(filters["sharpness"])))
    if "brightness" in filters:
        out = ImageEnhance.Brightness(out).enhance(max(0.1, 1.0 + float(filters["brightness"])))
    if "warmth" in filters:
        warmth = float(filters["warmth"])
        arr = np.array(out).astype(np.float32)
        arr[:, :, 0] *= 1.0 + max(0.0, warmth) * 0.12
        arr[:, :, 2] *= 1.0 + max(0.0, -warmth) * 0.12
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
    if "bloom" in filters:
        bloom = max(0.0, float(filters["bloom"]))
        if bloom > 0:
            blur = out.filter(ImageFilter.GaussianBlur(radius=max(0.25, bloom * 1.8)))
            out = Image.blend(out, blur, alpha=min(0.35, bloom * 0.45))
    if "grain" in filters:
        grain = max(0.0, float(filters["grain"]))
        if grain > 0:
            rng = np.random.default_rng(seed=seed if seed is not None else 42)
            arr = np.array(out).astype(np.float32)
            noise = rng.normal(loc=0.0, scale=255.0 * grain, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            out = Image.fromarray(arr)
    return out


def _reinforce_lineart(original: Image.Image, styled: Image.Image, strength: float = 1.0) -> Image.Image:
    strength = max(0.0, min(float(strength), 1.4))
    if strength <= 0:
        return styled
    src = np.array(original.convert("RGB"))
    dst = np.array(styled.convert("RGB"))
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    mask = (edges > 0).astype(np.float32)[:, :, None]
    alpha = min(0.55, 0.2 + 0.22 * strength)
    reinforced = dst.astype(np.float32)
    reinforced = reinforced * (1.0 - alpha * mask)
    reinforced = np.clip(reinforced, 0, 255).astype(np.uint8)
    return Image.fromarray(reinforced)


def _fallback_style(
    image: Image.Image,
    package: StylePackage,
    render_quality: str,
    seed: Optional[int],
) -> Image.Image:
    out = image.convert("RGB")
    # Quality-adaptive fast filters.
    if render_quality == "preview":
        out = out.filter(ImageFilter.SMOOTH)
    elif render_quality == "final":
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2))
    else:
        out = out.filter(ImageFilter.SMOOTH_MORE)
    out = _apply_post_filters(out, package=package, seed=seed)
    if package.preserve_line:
        out = _reinforce_lineart(image, out, strength=package.controlnet_config.get("lineart", 1.0))
    return out


def _diffusion_style(
    image: Image.Image,
    package: StylePackage,
    model_manager: ModelManager,
    seed: Optional[int],
    strength_override: Optional[float] = None,
    guidance_override: Optional[float] = None,
    steps_override: Optional[int] = None,
) -> Optional[Image.Image]:
    pipe = model_manager.load_img2img_pipeline()
    if pipe is None:
        return None
    strength = (
        float(strength_override)
        if strength_override is not None
        else float(package.img2img_params.get("strength", 0.45))
    )
    strength = max(0.15, min(strength, 0.8))
    steps = int(steps_override) if steps_override is not None else int(package.img2img_params.get("steps", 18))
    guidance = (
        float(guidance_override)
        if guidance_override is not None
        else float(package.img2img_params.get("guidance_scale", 7.5))
    )
    steps = max(6, min(90, int(steps)))
    guidance = max(1.0, min(14.0, float(guidance)))

    generator = None
    if seed is not None and model_manager.torch is not None:
        generator = model_manager.torch.Generator(model_manager.device).manual_seed(seed)

    try:
        result = pipe(
            prompt=package.prompt_template,
            negative_prompt=package.negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        result = result.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
        result = _apply_post_filters(result, package=package, seed=seed)
        if package.preserve_line:
            result = _reinforce_lineart(
                image,
                result,
                strength=package.controlnet_config.get("lineart", 1.0),
            )
        return result
    except Exception as exc:
        logger.warning("Diffusion style failed for {}: {}", package.id, exc)
        return None


def apply_style(
    image: Image.Image,
    style_name: str,
    model_manager: ModelManager,
    render_quality: str = "balanced",
    seed: Optional[int] = None,
    strength_override: Optional[float] = None,
    guidance_override: Optional[float] = None,
    steps_override: Optional[int] = None,
) -> Image.Image:
    result, _, _ = apply_style_with_report(
        image=image,
        style_name=style_name,
        model_manager=model_manager,
        render_quality=render_quality,
        seed=seed,
        strength_override=strength_override,
        guidance_override=guidance_override,
        steps_override=steps_override,
    )
    return result


def apply_style_with_report(
    image: Image.Image,
    style_name: str,
    model_manager: ModelManager,
    render_quality: str = "balanced",
    seed: Optional[int] = None,
    strength_override: Optional[float] = None,
    guidance_override: Optional[float] = None,
    steps_override: Optional[int] = None,
) -> tuple[Image.Image, str, str]:
    package = get_style_package(style_name, render_quality=render_quality, settings=model_manager.settings)
    if package.id == "original" and render_quality == "preview":
        return (
            _fallback_style(image, package=package, render_quality=render_quality, seed=seed),
            package.id,
            "fallback_filter",
        )

    result = _diffusion_style(
        image=image,
        package=package,
        model_manager=model_manager,
        seed=seed,
        strength_override=strength_override,
        guidance_override=guidance_override,
        steps_override=steps_override,
    )
    if result is not None:
        return result, package.id, "diffusion_img2img"

    # Reject silently degraded style results in strict/pro and all final-quality paths.
    strict_no_fallback = bool(
        render_quality in {"final", "quality"}
        or (model_manager.settings.strict_pro_mode and render_quality in {"balanced"})
    )
    if strict_no_fallback:
        raise RuntimeError(
            f"Strict mode blocked fallback style rendering for '{package.id}' due to missing diffusion pipeline."
        )
    return (
        _fallback_style(image, package=package, render_quality=render_quality, seed=seed),
        package.id,
        "fallback_filter",
    )


def apply_style_bytes(
    image_bytes: bytes,
    style_name: str,
    model_manager: ModelManager,
    render_quality: str = "balanced",
    seed: Optional[int] = None,
) -> bytes:
    image = safe_image_load(image_bytes)
    out = apply_style(
        image=image,
        style_name=style_name,
        model_manager=model_manager,
        render_quality=render_quality,
        seed=seed,
    )
    return pil_to_png_bytes(out)
