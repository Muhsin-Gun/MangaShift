from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.adaptive_controller import AdaptiveController  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.episodic_memory import EpisodicCharacterMemory  # noqa: E402
from app.repair_planner import build_repair_plan  # noqa: E402


class FakeCharacterDB:
    def detect_faces(self, image: Image.Image):
        # Deterministic fake face crops for testability.
        crop1 = image.crop((30, 30, 110, 110)).resize((160, 160))
        crop2 = image.crop((130, 30, 210, 110)).resize((160, 160))
        return [(crop1, (30, 30, 80, 80)), (crop2, (130, 30, 80, 80))]

    def embed_face(self, face: Image.Image):
        arr = np.array(face.convert("RGB"), dtype=np.float32)
        stats = np.array([arr.mean(), arr.std(), arr[:, :, 0].mean(), arr[:, :, 1].mean()], dtype=np.float32)
        stats = stats / (np.linalg.norm(stats) + 1e-9)
        return stats


def _sample_image() -> Image.Image:
    image = Image.new("RGB", (320, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 30, 110, 110), fill=(180, 140, 120))
    draw.ellipse((130, 30, 210, 110), fill=(120, 150, 190))
    return image


def test_repair_planner_actions():
    structural = {
        "edge_overlap": 0.41,
        "blur_score": 0.52,
        "contrast_shift": 48.0,
        "texture_shift": 0.14,
    }
    identity = {
        "best_similarity": 0.63,
        "threshold": 0.78,
        "drift_detected": True,
    }
    plan = build_repair_plan(structural, identity, render_quality="final", style_id="dark_fantasy")
    payload = plan.to_dict()
    assert payload["apply_rerender"] is True
    assert payload["strength_override"] <= 0.46
    assert len(payload["actions"]) >= 3


def test_episodic_memory_update(tmp_path: Path):
    memory_file = tmp_path / "episodic.json"
    em = EpisodicCharacterMemory(memory_path=memory_file, similarity_threshold=0.5, max_palette_colors=8)
    image = _sample_image()
    fake_db = FakeCharacterDB()

    update = em.update_from_panel(
        series_id="series_a",
        image=image,
        character_db=fake_db,  # type: ignore[arg-type]
        speaker_notes={"0": "formal captain voice", "1": "calm strategist"},
        page_index=1,
    )
    summary = em.summary("series_a")

    assert update["faces_detected"] == 2
    assert summary["characters_total"] >= 1
    assert len(summary["palette_colors"]) > 0
    assert memory_file.exists()


def test_adaptive_controller_decisions():
    settings = get_settings()
    controller = AdaptiveController(settings=settings)
    initial = controller.propose_initial(style_id="cinematic", render_quality="balanced", device="cpu")
    assert 0.2 <= initial.strength <= 0.72
    assert 5.0 <= initial.guidance_scale <= 11.0
    assert initial.steps >= 6

    retry = controller.propose_retry(
        style_id="cinematic",
        render_quality="balanced",
        device="cpu",
        structural_report={"edge_overlap": 0.4, "blur_score": 0.5, "contrast_shift": 45.0},
        identity_report={"best_similarity": 0.62, "threshold": 0.78, "drift_detected": True},
        attempt=1,
        previous=initial,
    )
    assert retry.strength <= initial.strength
    assert retry.steps >= initial.steps
