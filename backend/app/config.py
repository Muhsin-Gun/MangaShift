from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "MangaShift AI"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    models_dir: Path = Field(default=Path(__file__).resolve().parents[1] / "models")
    cache_dir: Path = Field(default=Path(__file__).resolve().parents[1] / "cache")
    loras_dir: Path = Field(default=Path(__file__).resolve().parents[1] / "loras")
    fonts_dir: Path = Field(default=Path(__file__).resolve().parents[1] / "fonts")
    style_packages_file: Path = Field(
        default=Path(__file__).resolve().parents[1] / "style_packages.json"
    )
    model_manifest_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "models" / "manifest.json"
    )

    max_cache_items: int = 512
    max_cache_size_gb: int = 10
    request_timeout_s: int = 120
    warmup_timeout_s: int = 300
    ocr_min_confidence: float = 0.35

    sd_inpaint_model: str = "runwayml/stable-diffusion-inpainting"
    sd_img2img_model: str = "runwayml/stable-diffusion-v1-5"
    realesrgan_model_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "models" / "realesrgan" / "RealESRGAN_x4plus.pth"
    )
    quality_sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_canny_model: str = "lllyasviel/sd-controlnet-canny"
    controlnet_depth_model: str = "lllyasviel/sd-controlnet-depth"
    controlnet_openpose_model: str = "lllyasviel/control_v11p_sd15_openpose"
    quality_controlnet_canny_model: str = "diffusers/controlnet-canny-sdxl-1.0"
    quality_controlnet_depth_model: str = "diffusers/controlnet-depth-sdxl-1.0"
    quality_controlnet_openpose_model: str = "thibaud/controlnet-openpose-sdxl-1.0"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.safetensors"
    quality_ip_adapter_repo: str = "h94/IP-Adapter"
    quality_ip_adapter_subfolder: str = "sdxl_models"
    quality_ip_adapter_weight_name: str = "ip-adapter_sdxl_vit-h.safetensors"
    marian_ja_en: str = "Helsinki-NLP/opus-mt-ja-en"
    marian_ko_en: str = "Helsinki-NLP/opus-mt-ko-en"
    m2m100_fallback: str = "facebook/m2m100_418M"

    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    enable_diffusion_styles: bool = True
    enable_diffusion_inpaint: bool = True
    enable_llm_post_edit: bool = False
    enable_scene_memory: bool = True
    enable_episodic_memory: bool = True
    enable_repair_planner: bool = True
    enable_adaptive_controller: bool = True
    enforce_identity_consistency: bool = True
    local_models_only: bool = True
    strict_pro_mode: bool = False
    strict_require_gpu: bool = False
    strict_require_diffusion: bool = False
    strict_require_ocr: bool = False
    strict_require_translation_models: bool = False
    quality_require_gpu: bool = True
    quality_require_full_path: bool = True
    quality_cloud_worker_url: str = ""
    quality_cloud_worker_token: str = ""
    identity_similarity_threshold: float = 0.78
    identity_max_retries: int = 2
    quality_identity_similarity_threshold: float = 0.78
    quality_style_adherence_threshold: float = 0.64
    style_line_weight_default: float = 1.0
    style_hatch_density_default: float = 0.52
    style_tone_intensity_default: float = 0.42
    style_screentone_strength_default: float = 0.16
    translation_context_window_pages: int = 6
    translation_context_window_lines: int = 48
    quality_preview_steps: int = 8
    quality_balanced_steps: int = 18
    quality_final_steps: int = 30
    quality_master_steps: int = 42
    quality_variant_min: int = 16
    quality_variant_max: int = 32
    quality_topk: int = 3
    default_variant_count: int = 6
    controller_target_identity: float = 0.82
    controller_target_edge_overlap: float = 0.64
    controller_target_blur_score: float = 0.72
    episodic_similarity_threshold: float = 0.78
    episodic_max_palette_colors: int = 12
    episodic_memory_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "cache" / "episodic_memory.json"
    )
    llm_gguf_path: Path = Field(default=Path(__file__).resolve().parents[1] / "models" / "llm" / "translator.gguf")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.fonts_dir.mkdir(parents=True, exist_ok=True)
    settings.loras_dir.mkdir(parents=True, exist_ok=True)
    settings.style_packages_file.parent.mkdir(parents=True, exist_ok=True)
    settings.model_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.episodic_memory_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
