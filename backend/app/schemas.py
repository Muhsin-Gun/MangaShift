from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(SchemaBase):
    status: str = "ok"
    time_ms: int
    device: str
    strict_mode: bool = False


class ProcessOptions(SchemaBase):
    source_lang: str = Field(default="auto")
    target_lang: str = Field(default="en")
    style: str = Field(default="original")
    colorize: bool = Field(default=False)
    upscale: float = Field(default=1.0, ge=1.0, le=4.0)
    series_id: Optional[str] = None
    chapter_id: Optional[str] = None
    page_index: int = Field(default=0, ge=0)
    render_quality: str = Field(default="balanced")
    line_weight: float = Field(default=1.0, ge=0.5, le=2.0)
    hatch_density: float = Field(default=0.52, ge=0.1, le=1.0)
    tone_intensity: float = Field(default=0.42, ge=0.0, le=1.0)
    screentone_strength: float = Field(default=0.16, ge=0.0, le=1.0)
    enforce_identity: bool = Field(default=True)
    priority: int = Field(default=5, ge=1, le=10)
    inpaint_padding: int = Field(default=8, ge=0, le=64)


class ProcessReport(SchemaBase):
    cached: bool
    image_width: int
    image_height: int
    ocr_regions: int
    translated_regions: int
    style_id: str = "original"
    render_quality: str = "balanced"
    missing_capabilities: list[str] = Field(default_factory=list)
    identity_report: dict = Field(default_factory=dict)
    structural_report: dict = Field(default_factory=dict)
    planner_report: dict = Field(default_factory=dict)
    episodic_report: dict = Field(default_factory=dict)
    engine_report: dict = Field(default_factory=dict)
    timings_ms: dict


class PreflightResponse(SchemaBase):
    app: str
    python: dict
    device: str
    gpu: dict
    strict_mode: bool
    strict_flags: dict
    feature_flags: dict
    local_models_only: bool
    modules: dict
    model_paths: dict
    model_manifest: dict = Field(default_factory=dict)
    tier: str
    readiness_score: int
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


class EpisodicSummaryResponse(SchemaBase):
    series_id: str
    characters_total: int
    seen_faces_total: int
    palette_colors: list[str] = Field(default_factory=list)
    voice_profile: dict = Field(default_factory=dict)
    updated_at: str = ""


class WarmupRequest(SchemaBase):
    render_quality: str = Field(default="balanced")
    style: str = Field(default="original")
    source_lang: str = Field(default="auto")
    colorize: bool = Field(default=False)
    include_llm: bool = Field(default=False)
    strict: bool = Field(default=False)


class WarmupResponse(SchemaBase):
    status: str
    ok: bool
    device: str
    strict_mode: bool
    requested: dict = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    steps: list[dict] = Field(default_factory=list)
    timings_ms: dict = Field(default_factory=dict)
    capabilities: dict = Field(default_factory=dict)
