from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.quality_gate import QualityThresholds, evaluate_quality_gate  # noqa: E402


def _build_line_image() -> Image.Image:
    img = Image.new("RGB", (320, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.line((20, 20, 300, 20), fill="black", width=2)
    draw.line((20, 60, 300, 300), fill="black", width=2)
    draw.line((30, 290, 290, 40), fill="black", width=2)
    return img


def test_quality_gate_contains_line_metrics():
    original = _build_line_image()
    structure = _build_line_image()
    rendered = _build_line_image()
    report = evaluate_quality_gate(
        original=original,
        structure_pass=structure,
        rendered=rendered,
        thresholds=QualityThresholds(),
        use_lpips_model=False,
        style_reference=None,
        identity_reference=original,
        character_db=None,
    )
    payload = report.to_dict()
    assert "line_sharpness" in payload
    assert "line_continuity" in payload
    assert "worm_artifact_score" in payload
    assert isinstance(payload["line_sharpness"], float)
    assert isinstance(payload["line_continuity"], float)
    assert isinstance(payload["worm_artifact_score"], float)


def test_quality_gate_can_fail_when_identity_faces_missing():
    original = _build_line_image()
    structure = _build_line_image()
    rendered = _build_line_image()
    report = evaluate_quality_gate(
        original=original,
        structure_pass=structure,
        rendered=rendered,
        thresholds=QualityThresholds(),
        use_lpips_model=False,
        style_reference=None,
        identity_reference=original,
        character_db=None,
        fail_on_missing_faces=True,
    )
    payload = report.to_dict()
    assert payload["passed"] is False
    assert "identity_faces_missing" in payload["reasons"]
