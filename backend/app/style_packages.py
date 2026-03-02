from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .config import Settings


QUALITY_STEPS = {
    "preview": 8,
    "balanced": 18,
    "final": 30,
    "quality": 42,
}

LEGACY_STYLE_ALIASES = {
    "shonen": "shonen_action",
    "shoujo": "shoujo_soft",
    "gritty": "seinen_gritty",
    "painterly": "painterly_watercolor",
    "cinematic": "cinematic_filmic",
    "retro_manga": "retro_90s",
    "chibi": "chibi_cute",
}


@dataclass
class StylePackage:
    id: str
    name: str
    short_desc: str
    thumbnail: str
    prompt_template: str
    negative_prompt: str
    controlnet_config: Dict[str, float]
    img2img_params: Dict[str, float]
    lora_reference: Optional[str] = None
    palette_anchor: List[str] = field(default_factory=list)
    post_filters: Dict[str, float] = field(default_factory=dict)
    preserve_line: bool = True
    priority: str = "balanced"

    def normalized(self) -> "StylePackage":
        cleaned = copy.deepcopy(self)
        cleaned.id = cleaned.id.strip().lower().replace(" ", "_")
        cleaned.priority = (
            cleaned.priority
            if cleaned.priority in {"preview", "balanced", "final", "quality"}
            else "balanced"
        )
        strength = float(cleaned.img2img_params.get("strength", 0.45))
        cleaned.img2img_params["strength"] = max(0.15, min(strength, 0.8))
        steps = int(cleaned.img2img_params.get("steps", QUALITY_STEPS[cleaned.priority]))
        cleaned.img2img_params["steps"] = max(4, min(steps, 80))
        guidance = float(cleaned.img2img_params.get("guidance_scale", 7.5))
        cleaned.img2img_params["guidance_scale"] = max(1.0, min(guidance, 20.0))
        return cleaned

    def with_quality(self, quality: str, settings: Settings) -> "StylePackage":
        package = self.normalized()
        if quality not in {"preview", "balanced", "final", "quality"}:
            quality = "balanced"
        if quality == "preview":
            steps = settings.quality_preview_steps
            package.img2img_params["strength"] = min(package.img2img_params["strength"], 0.42)
        elif quality == "final":
            steps = settings.quality_final_steps
            package.img2img_params["strength"] = min(max(package.img2img_params["strength"], 0.4), 0.72)
        elif quality == "quality":
            steps = settings.quality_master_steps
            package.img2img_params["strength"] = min(max(package.img2img_params["strength"], 0.5), 0.75)
            package.img2img_params["guidance_scale"] = min(
                max(package.img2img_params.get("guidance_scale", 8.5), 8.5),
                12.0,
            )
        else:
            steps = settings.quality_balanced_steps
        package.img2img_params["steps"] = max(4, min(int(steps), 90))
        return package

    @classmethod
    def from_dict(cls, raw: dict) -> "StylePackage":
        return cls(
            id=str(raw.get("id", "original")),
            name=str(raw.get("name", "Original")),
            short_desc=str(raw.get("short_desc", "")),
            thumbnail=str(raw.get("thumbnail", "")),
            prompt_template=str(raw.get("prompt_template", "")),
            negative_prompt=str(
                raw.get(
                    "negative_prompt",
                    "blurry, extra limbs, bad anatomy, watermark, text artifacts",
                )
            ),
            controlnet_config=dict(raw.get("controlnet_config", {})),
            img2img_params=dict(raw.get("img2img_params", {})),
            lora_reference=raw.get("lora_reference"),
            palette_anchor=list(raw.get("palette_anchor", [])),
            post_filters=dict(raw.get("post_filters", {})),
            preserve_line=bool(raw.get("preserve_line", True)),
            priority=str(raw.get("priority", "balanced")),
        ).normalized()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "short_desc": self.short_desc,
            "thumbnail": self.thumbnail,
            "prompt_template": self.prompt_template,
            "negative_prompt": self.negative_prompt,
            "controlnet_config": self.controlnet_config,
            "img2img_params": self.img2img_params,
            "lora_reference": self.lora_reference,
            "palette_anchor": self.palette_anchor,
            "post_filters": self.post_filters,
            "preserve_line": self.preserve_line,
            "priority": self.priority,
        }


class StylePackageManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._styles: Dict[str, StylePackage] = {}
        self._load_defaults()
        self._load_user_packages(settings.style_packages_file)

    def _load_defaults(self) -> None:
        defaults = [StylePackage.from_dict(item) for item in _default_styles()]
        self._styles = {pkg.id: pkg for pkg in defaults}

    def _load_user_packages(self, source_file: Path) -> None:
        if not source_file.exists():
            return
        try:
            payload = json.loads(source_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading style package file {}: {}", source_file, exc)
            return
        if not isinstance(payload, list):
            logger.warning("Style package file {} is not a list", source_file)
            return
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            pkg = StylePackage.from_dict(raw)
            self._styles[pkg.id] = pkg

    def list_ids(self) -> List[str]:
        return sorted(self._styles.keys())

    def list_dicts(self) -> List[dict]:
        return [self._styles[k].to_dict() for k in self.list_ids()]

    def get(self, style_id: str, quality: str = "balanced") -> StylePackage:
        key = (style_id or "original").strip().lower()
        key = LEGACY_STYLE_ALIASES.get(key, key)
        package = self._styles.get(key) or self._styles["original"]
        return package.with_quality(quality, self.settings)

    def exists(self, style_id: str) -> bool:
        key = style_id.strip().lower()
        key = LEGACY_STYLE_ALIASES.get(key, key)
        return key in self._styles

    def save_user_packages(self, packages: List[StylePackage]) -> None:
        serializable = [pkg.to_dict() for pkg in packages]
        self.settings.style_packages_file.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._load_defaults()
        self._load_user_packages(self.settings.style_packages_file)


def _default_styles() -> List[dict]:
    base_negative = "blurry, low quality, malformed anatomy, extra limbs, watermark, text artifacts, jpeg artifacts"
    return [
        {
            "id": "original",
            "name": "Original / Faithful",
            "short_desc": "Preserve original lines and tones with minimal changes.",
            "thumbnail": "assets/style_original.png",
            "prompt_template": "faithful manga restoration, preserve original composition and line art",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.1, "depth": 0.0, "openpose": 0.0},
            "img2img_params": {"strength": 0.2, "steps": 10, "guidance_scale": 5.5},
            "palette_anchor": ["#f6f6f6", "#2b2b2b"],
            "post_filters": {"contrast": 0.04, "sharpness": 0.08},
            "preserve_line": True,
            "priority": "preview",
        },
        {
            "id": "webtoon_smooth",
            "name": "Webtoon Smooth",
            "short_desc": "Flat smooth digital shading with bright readable tones.",
            "thumbnail": "assets/style_webtoon_smooth.png",
            "prompt_template": "korean webtoon style, clean cel shading, smooth color fill, clear linework",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.0, "depth": 0.2, "openpose": 0.2},
            "img2img_params": {"strength": 0.42, "steps": 18, "guidance_scale": 7.1},
            "palette_anchor": ["#ffd9c8", "#4f5c6b", "#c9d7ef"],
            "post_filters": {"saturation": 0.12, "contrast": 0.07, "sharpness": 0.06},
            "preserve_line": True,
            "priority": "balanced",
        },
        {
            "id": "shonen_action",
            "name": "Shonen Action",
            "short_desc": "Bold action shading and high-contrast impact.",
            "thumbnail": "assets/style_shonen_action.png",
            "prompt_template": "dynamic shonen action manga, bold ink lines, cinematic highlights, explosive motion",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.05, "depth": 0.25, "openpose": 0.35},
            "img2img_params": {"strength": 0.5, "steps": 24, "guidance_scale": 8.0},
            "palette_anchor": ["#ffb347", "#243447", "#e8edf2"],
            "post_filters": {"contrast": 0.16, "saturation": 0.1, "sharpness": 0.12},
            "preserve_line": True,
            "priority": "final",
        },
        {
            "id": "shoujo_soft",
            "name": "Shoujo Soft",
            "short_desc": "Delicate romance shading with pastel gradients.",
            "thumbnail": "assets/style_shoujo_soft.png",
            "prompt_template": "shoujo romance manga style, soft glow, elegant linework, pastel tones",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.95, "depth": 0.15, "openpose": 0.1},
            "img2img_params": {"strength": 0.46, "steps": 22, "guidance_scale": 7.4},
            "palette_anchor": ["#ffd9e6", "#f8d1c8", "#e9edf9"],
            "post_filters": {"brightness": 0.06, "saturation": 0.1, "bloom": 0.1},
            "preserve_line": True,
            "priority": "balanced",
        },
        {
            "id": "seinen_gritty",
            "name": "Seinen / Gritty",
            "short_desc": "Darker grounded texture with heavy contrast.",
            "thumbnail": "assets/style_seinen_gritty.png",
            "prompt_template": "seinen manga style, gritty texture, dramatic shadows, realistic line weight",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.1, "depth": 0.35, "openpose": 0.15},
            "img2img_params": {"strength": 0.54, "steps": 26, "guidance_scale": 8.3},
            "palette_anchor": ["#2e2f33", "#8d8f95", "#d9dbe1"],
            "post_filters": {"contrast": 0.2, "saturation": -0.06, "grain": 0.06},
            "preserve_line": True,
            "priority": "final",
        },
        {
            "id": "ultra_realistic",
            "name": "Ultra-Realistic",
            "short_desc": "High-detail realistic rendering with cinematic depth.",
            "thumbnail": "assets/style_ultra_realistic.png",
            "prompt_template": (
                "ultra realistic illustration, detailed skin textures, "
                "global illumination, cinematic scene"
            ),
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.9, "depth": 0.6, "openpose": 0.4},
            "img2img_params": {"strength": 0.62, "steps": 32, "guidance_scale": 8.9},
            "palette_anchor": ["#f0d6c2", "#4a4d53", "#8798b0"],
            "post_filters": {"contrast": 0.12, "sharpness": 0.16, "grain": 0.04},
            "preserve_line": False,
            "priority": "final",
        },
        {
            "id": "realistic",
            "name": "Realistic",
            "short_desc": "Balanced realistic rendering with preserved readability.",
            "thumbnail": "assets/style_realistic.png",
            "prompt_template": "realistic manga redraw, natural skin tones, cinematic shadows, coherent anatomy",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.95, "depth": 0.45, "openpose": 0.3},
            "img2img_params": {"strength": 0.53, "steps": 26, "guidance_scale": 8.0},
            "palette_anchor": ["#efcfb9", "#4a423a", "#b19f8b"],
            "post_filters": {"contrast": 0.1, "sharpness": 0.14, "grain": 0.03},
            "preserve_line": True,
            "priority": "final",
        },
        {
            "id": "painterly_watercolor",
            "name": "Painterly / Watercolor",
            "short_desc": "Soft brush texture and watercolor blending.",
            "thumbnail": "assets/style_painterly_watercolor.png",
            "prompt_template": (
                "watercolor anime illustration, painterly brushwork, "
                "flowing pigments, expressive textures"
            ),
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.85, "depth": 0.25, "openpose": 0.1},
            "img2img_params": {"strength": 0.56, "steps": 25, "guidance_scale": 7.6},
            "palette_anchor": ["#c5d4d8", "#e2b08c", "#6f7f8a"],
            "post_filters": {"saturation": 0.07, "contrast": 0.03, "bloom": 0.08},
            "preserve_line": False,
            "priority": "balanced",
        },
        {
            "id": "cinematic_filmic",
            "name": "Cinematic",
            "short_desc": "Filmic grading, rim light, and atmosphere.",
            "thumbnail": "assets/style_cinematic_filmic.png",
            "prompt_template": "cinematic anime frame, filmic color grading, volumetric light, atmospheric depth",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.0, "depth": 0.45, "openpose": 0.2},
            "img2img_params": {"strength": 0.5, "steps": 28, "guidance_scale": 8.2},
            "palette_anchor": ["#d8c29c", "#455164", "#b8c0d2"],
            "post_filters": {"contrast": 0.18, "warmth": -0.06, "grain": 0.05},
            "preserve_line": True,
            "priority": "final",
        },
        {
            "id": "noir_contrast",
            "name": "Noir / High-Contrast",
            "short_desc": "Strong monochrome chiaroscuro with punchy blacks.",
            "thumbnail": "assets/style_noir_contrast.png",
            "prompt_template": "noir manga panel, dramatic black and white contrast, chiaroscuro composition",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.2, "depth": 0.1, "openpose": 0.0},
            "img2img_params": {"strength": 0.34, "steps": 20, "guidance_scale": 6.8},
            "palette_anchor": ["#0c0c0c", "#f1f1f1"],
            "post_filters": {"contrast": 0.26, "saturation": -0.9, "grain": 0.02},
            "preserve_line": True,
            "priority": "balanced",
        },
        {
            "id": "pastel_soft_glow",
            "name": "Pastel / Soft-Glow",
            "short_desc": "Low-contrast pastel ambience and soft bloom.",
            "thumbnail": "assets/style_pastel_soft_glow.png",
            "prompt_template": "pastel anime palette, soft glow, dreamy shading, clean line details",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.88, "depth": 0.12, "openpose": 0.1},
            "img2img_params": {"strength": 0.44, "steps": 20, "guidance_scale": 6.9},
            "palette_anchor": ["#ffd6e5", "#f8ecd8", "#cfd9f7"],
            "post_filters": {"brightness": 0.08, "saturation": 0.05, "bloom": 0.15},
            "preserve_line": True,
            "priority": "balanced",
        },
        {
            "id": "dark_fantasy",
            "name": "Dark Fantasy",
            "short_desc": "Heavy atmosphere, deep shadows, and ominous tones.",
            "thumbnail": "assets/style_dark_fantasy.png",
            "prompt_template": "dark fantasy manga concept art, moody atmosphere, deep shadows, volumetric haze",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.05, "depth": 0.6, "openpose": 0.2},
            "img2img_params": {"strength": 0.58, "steps": 30, "guidance_scale": 8.8},
            "palette_anchor": ["#2b2235", "#5a516e", "#d5c9b0"],
            "post_filters": {"contrast": 0.2, "saturation": -0.12, "grain": 0.07},
            "preserve_line": True,
            "priority": "final",
        },
        {
            "id": "manga_classic",
            "name": "Manga Classic",
            "short_desc": "Crisp line enhancement with screentone-friendly output.",
            "thumbnail": "assets/style_manga_classic.png",
            "prompt_template": "classic manga lineart, screentone style shading, clean inks, faithful panel structure",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.15, "depth": 0.05, "openpose": 0.0},
            "img2img_params": {"strength": 0.28, "steps": 16, "guidance_scale": 5.9},
            "palette_anchor": ["#f5f5f5", "#1d1d1d", "#9b9b9b"],
            "post_filters": {"contrast": 0.14, "sharpness": 0.14},
            "preserve_line": True,
            "priority": "preview",
        },
        {
            "id": "painterly_realism",
            "name": "Painterly Realism",
            "short_desc": "Glossy high-end webtoon realism with painterly detail.",
            "thumbnail": "assets/style_painterly_realism.png",
            "prompt_template": "high-end korean webtoon realism, glossy skin, nuanced cloth folds, cinematic tones",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.96, "depth": 0.52, "openpose": 0.3},
            "img2img_params": {"strength": 0.6, "steps": 32, "guidance_scale": 8.7},
            "palette_anchor": ["#edc7ac", "#2f3948", "#9ba9bc"],
            "post_filters": {"contrast": 0.15, "sharpness": 0.18, "grain": 0.03},
            "preserve_line": False,
            "priority": "final",
        },
        {
            "id": "manager_drama",
            "name": "Manager Drama",
            "short_desc": "Clean romance-office style with warm highlights.",
            "thumbnail": "assets/style_manager_drama.png",
            "prompt_template": (
                "romance office webtoon style, clean facial rendering, "
                "subtle shading, warm cinematic tones"
            ),
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.92, "depth": 0.3, "openpose": 0.2},
            "img2img_params": {"strength": 0.48, "steps": 24, "guidance_scale": 7.2},
            "palette_anchor": ["#f3d2bf", "#5a6573", "#f0e5d2"],
            "post_filters": {"warmth": 0.09, "contrast": 0.08, "saturation": 0.06},
            "preserve_line": True,
            "priority": "balanced",
        },
        {
            "id": "retro_90s",
            "name": "Retro / 90s Manga",
            "short_desc": "Old-school contrast and print-grain feel.",
            "thumbnail": "assets/style_retro_90s.png",
            "prompt_template": "1990s manga print style, halftone texture, retro contrast, vintage paper mood",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.1, "depth": 0.08, "openpose": 0.05},
            "img2img_params": {"strength": 0.36, "steps": 18, "guidance_scale": 6.4},
            "palette_anchor": ["#f2e8d5", "#2d2a25", "#b8ab95"],
            "post_filters": {"contrast": 0.17, "grain": 0.1, "warmth": 0.05},
            "preserve_line": True,
            "priority": "balanced",
        },
        {
            "id": "chibi_cute",
            "name": "Chibi / Cute",
            "short_desc": "Rounded cute style with pastel pop colors.",
            "thumbnail": "assets/style_chibi_cute.png",
            "prompt_template": "cute chibi anime style, rounded forms, bright pastel colors, playful rendering",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.8, "depth": 0.1, "openpose": 0.35},
            "img2img_params": {"strength": 0.62, "steps": 26, "guidance_scale": 8.6},
            "palette_anchor": ["#ffd3de", "#ffe7a5", "#9fd9f8"],
            "post_filters": {"saturation": 0.22, "contrast": 0.05, "bloom": 0.12},
            "preserve_line": False,
            "priority": "balanced",
        },
        {
            "id": "hybrid_3d_cel",
            "name": "3D Hybrid / Cel-Shaded",
            "short_desc": "Subtle 3D volume with anime cel line finishing.",
            "thumbnail": "assets/style_hybrid_3d_cel.png",
            "prompt_template": "hybrid 3d cel-shaded anime look, structured lighting, crisp outlines",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.9, "depth": 0.55, "openpose": 0.25},
            "img2img_params": {"strength": 0.57, "steps": 28, "guidance_scale": 8.1},
            "palette_anchor": ["#e0c7b5", "#39485e", "#c9d4e4"],
            "post_filters": {"contrast": 0.12, "sharpness": 0.1, "saturation": 0.03},
            "preserve_line": True,
            "priority": "final",
        },
        {
            "id": "painterly_fantasy",
            "name": "Painterly Fantasy",
            "short_desc": "Epic fantasy color lighting with brush texture.",
            "thumbnail": "assets/style_painterly_fantasy.png",
            "prompt_template": (
                "epic fantasy anime painting, dramatic light rays, "
                "rich brush textures, magical atmosphere"
            ),
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 0.9, "depth": 0.65, "openpose": 0.2},
            "img2img_params": {"strength": 0.64, "steps": 34, "guidance_scale": 9.0},
            "palette_anchor": ["#c9b5ff", "#40507a", "#ffd8a2"],
            "post_filters": {"contrast": 0.14, "saturation": 0.16, "bloom": 0.14},
            "preserve_line": False,
            "priority": "final",
        },
        {
            "id": "ink_art",
            "name": "Ink Art",
            "short_desc": "High-impact ink brush style with strong edges.",
            "thumbnail": "assets/style_ink_art.png",
            "prompt_template": "ink brush manga style, bold strokes, textured black ink, high contrast line art",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.18, "depth": 0.1, "openpose": 0.02},
            "img2img_params": {"strength": 0.3, "steps": 15, "guidance_scale": 6.1},
            "palette_anchor": ["#fafafa", "#111111"],
            "post_filters": {"contrast": 0.24, "sharpness": 0.2},
            "preserve_line": True,
            "priority": "preview",
        },
        {
            "id": "sketch",
            "name": "Sketch",
            "short_desc": "Graphite-like texture and rough pencil feel.",
            "thumbnail": "assets/style_sketch.png",
            "prompt_template": "pencil sketch anime draft, graphite shading, expressive rough linework",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.12, "depth": 0.03, "openpose": 0.0},
            "img2img_params": {"strength": 0.27, "steps": 14, "guidance_scale": 5.7},
            "palette_anchor": ["#f1f1f1", "#303030", "#8d8d8d"],
            "post_filters": {"contrast": 0.18, "saturation": -0.95, "grain": 0.08},
            "preserve_line": True,
            "priority": "preview",
        },
        {
            "id": "smooth",
            "name": "Smooth Anime",
            "short_desc": "Balanced clean anime look for daily reading.",
            "thumbnail": "assets/style_smooth.png",
            "prompt_template": "smooth anime render, clean cel shading, faithful composition and color clarity",
            "negative_prompt": base_negative,
            "controlnet_config": {"lineart": 1.0, "depth": 0.2, "openpose": 0.2},
            "img2img_params": {"strength": 0.43, "steps": 20, "guidance_scale": 7.2},
            "palette_anchor": ["#f0d3c5", "#39495f", "#d6dfef"],
            "post_filters": {"contrast": 0.08, "saturation": 0.07, "sharpness": 0.08},
            "preserve_line": True,
            "priority": "balanced",
        },
    ]
