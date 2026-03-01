from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .character_db import CharacterDatabase
from .config import Settings


@dataclass
class IdentityReport:
    enabled: bool
    attempts: int
    faces_compared: int
    best_similarity: float
    threshold: float
    drift_detected: bool
    character_matches: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "attempts": self.attempts,
            "faces_compared": self.faces_compared,
            "best_similarity": round(self.best_similarity, 6),
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "character_matches": self.character_matches,
            "note": self.note,
        }


class IdentityEnforcer:
    def __init__(self, settings: Settings, character_db: CharacterDatabase):
        self.settings = settings
        self.character_db = character_db

    def _match_face_indices(
        self,
        source_boxes: List[Tuple[int, int, int, int]],
        target_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        used_target = set()
        for src_idx, src in enumerate(source_boxes):
            sx, sy, sw, sh = src
            src_center = np.array([sx + sw / 2.0, sy + sh / 2.0], dtype=np.float32)
            best_target = None
            best_dist = float("inf")
            for tgt_idx, tgt in enumerate(target_boxes):
                if tgt_idx in used_target:
                    continue
                tx, ty, tw, th = tgt
                tgt_center = np.array([tx + tw / 2.0, ty + th / 2.0], dtype=np.float32)
                dist = float(np.linalg.norm(src_center - tgt_center))
                if dist < best_dist:
                    best_dist = dist
                    best_target = tgt_idx
            if best_target is not None:
                pairs.append((src_idx, best_target))
                used_target.add(best_target)
        return pairs

    def _similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-9)
        emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-9)
        return float(np.dot(emb_a, emb_b))

    def evaluate(
        self,
        original: Image.Image,
        rendered: Image.Image,
        series_id: str,
    ) -> Tuple[float, int, int]:
        src_faces = self.character_db.detect_faces(original)
        dst_faces = self.character_db.detect_faces(rendered)
        if not src_faces or not dst_faces:
            return 1.0, 0, 0

        src_boxes = [bbox for _, bbox in src_faces]
        dst_boxes = [bbox for _, bbox in dst_faces]
        pairs = self._match_face_indices(src_boxes, dst_boxes)
        if not pairs:
            return 1.0, 0, 0

        scores: List[float] = []
        char_matches = 0
        for src_idx, dst_idx in pairs:
            src_img, _ = src_faces[src_idx]
            dst_img, _ = dst_faces[dst_idx]
            src_emb = self.character_db.embed_face(src_img)
            dst_emb = self.character_db.embed_face(dst_img)
            scores.append(self._similarity(src_emb, dst_emb))
            src_char = self.character_db.identify(series_id, src_img, threshold=0.55)
            dst_char = self.character_db.identify(series_id, dst_img, threshold=0.55)
            if src_char and dst_char and src_char == dst_char:
                char_matches += 1
        if not scores:
            return 1.0, 0, 0
        return float(sum(scores) / len(scores)), len(scores), char_matches

    def enforce(
        self,
        original: Image.Image,
        initial_render: Image.Image,
        series_id: Optional[str],
        rerender_callback: Callable[[int, float], Image.Image],
        force_enabled: bool = True,
    ) -> Tuple[Image.Image, IdentityReport]:
        enabled = bool(force_enabled and self.settings.enforce_identity_consistency and series_id)
        if not enabled:
            return initial_render, IdentityReport(
                enabled=False,
                attempts=0,
                faces_compared=0,
                best_similarity=1.0,
                threshold=self.settings.identity_similarity_threshold,
                drift_detected=False,
                character_matches=0,
                note="Identity consistency disabled",
            )

        best_image = initial_render
        best_score, compared, char_matches = self.evaluate(original, best_image, series_id=series_id or "")
        attempts = 0
        for attempt in range(1, self.settings.identity_max_retries + 1):
            if best_score >= self.settings.identity_similarity_threshold:
                break
            attempts = attempt
            candidate = rerender_callback(attempt, best_score)
            score, compared_now, char_matches_now = self.evaluate(original, candidate, series_id=series_id or "")
            if score > best_score:
                best_score = score
                best_image = candidate
                compared = compared_now
                char_matches = char_matches_now

        drift_detected = bool(best_score < self.settings.identity_similarity_threshold and compared > 0)
        report = IdentityReport(
            enabled=True,
            attempts=attempts,
            faces_compared=compared,
            best_similarity=best_score,
            threshold=self.settings.identity_similarity_threshold,
            drift_detected=drift_detected,
            character_matches=char_matches,
            note="Identity rerender loop applied" if attempts else "Identity check passed on first pass",
        )
        return best_image, report
