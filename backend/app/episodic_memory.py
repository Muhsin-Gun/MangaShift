from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .character_db import CharacterDatabase


_TOKEN_RE = re.compile(r"[a-zA-Z']{3,}")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


class EpisodicCharacterMemory:
    """Series-level episodic character memory with identity, palette, and voice priors."""

    def __init__(
        self,
        memory_path: Path,
        similarity_threshold: float = 0.78,
        max_palette_colors: int = 12,
        max_voice_tokens: int = 24,
    ):
        self.memory_path = memory_path
        self.similarity_threshold = float(similarity_threshold)
        self.max_palette_colors = int(max_palette_colors)
        self.max_voice_tokens = int(max_voice_tokens)
        self._lock = threading.Lock()
        self._memory: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.memory_path.exists():
            self._memory = {}
            return
        try:
            self._memory = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except Exception:
            self._memory = {}

    def _save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(
            json.dumps(self._memory, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def _series(self, series_id: str) -> dict:
        key = (series_id or "default_series").strip() or "default_series"
        if key not in self._memory:
            self._memory[key] = {
                "characters": {},
                "seen_faces_total": 0,
                "updated_at": "",
            }
        return self._memory[key]

    def _next_char_id(self, series_state: dict) -> str:
        chars = series_state.get("characters", {})
        n = len(chars) + 1
        while f"char_{n}" in chars:
            n += 1
        return f"char_{n}"

    def _extract_palette(self, face: Image.Image, max_colors: int = 4) -> List[str]:
        arr = np.array(face.convert("RGB").resize((24, 24), Image.Resampling.BILINEAR), dtype=np.uint8)
        flat = arr.reshape(-1, 3)
        quant = (flat // 32) * 32
        unique, counts = np.unique(quant, axis=0, return_counts=True)
        order = np.argsort(-counts)
        colors: List[str] = []
        for idx in order[:max_colors]:
            r, g, b = [int(v) for v in unique[idx]]
            colors.append(f"#{r:02x}{g:02x}{b:02x}")
        return colors

    def _merge_palette(self, existing: List[str], new_values: List[str]) -> List[str]:
        merged = list(existing)
        for color in new_values:
            if color not in merged:
                merged.append(color)
        return merged[: self.max_palette_colors]

    def _voice_tokens(self, note: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for token in _TOKEN_RE.findall((note or "").lower()):
            out[token] = out.get(token, 0) + 1
        return out

    def _merge_voice_tokens(self, existing: Dict[str, int], update: Dict[str, int]) -> Dict[str, int]:
        merged = dict(existing)
        for token, count in update.items():
            merged[token] = int(merged.get(token, 0)) + int(count)
        ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[: self.max_voice_tokens]
        return {k: int(v) for k, v in ranked}

    def _expression_hint(self, face: Image.Image) -> str:
        gray = np.array(face.convert("L"), dtype=np.float32)
        mean = float(gray.mean())
        std = float(gray.std())
        if std > 62.0:
            return "intense"
        if mean < 70.0:
            return "dark"
        if mean > 185.0:
            return "bright"
        return "neutral"

    def _pose_prior(self, bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> Dict[str, float]:
        x, y, w, h = [float(v) for v in bbox]
        iw, ih = [max(1.0, float(v)) for v in image_size]
        aspect = w / max(1.0, h)
        return {
            "x": x / iw,
            "y": y / ih,
            "w": w / iw,
            "h": h / ih,
            "aspect": aspect,
        }

    def _update_pose_prior(
        self,
        existing: Dict[str, float],
        update: Dict[str, float],
        alpha: float = 0.2,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in ("x", "y", "w", "h", "aspect"):
            old = float(existing.get(key, update.get(key, 0.0)))
            new = float(update.get(key, old))
            out[key] = (1.0 - alpha) * old + alpha * new
        return out

    def upsert_face(
        self,
        series_id: str,
        face: Image.Image,
        bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        character_db: CharacterDatabase,
        speaker_note: str = "",
        page_index: int = 0,
    ) -> Tuple[str, float, bool]:
        series = self._series(series_id)
        chars: Dict[str, dict] = series.setdefault("characters", {})
        emb = character_db.embed_face(face)

        best_id: Optional[str] = None
        best_sim = -1.0
        for char_id, data in chars.items():
            centroid = np.array(data.get("centroid", []), dtype=np.float32)
            if centroid.size == 0:
                continue
            sim = _cosine(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = char_id

        created = False
        if best_id is None or best_sim < self.similarity_threshold:
            best_id = self._next_char_id(series)
            chars[best_id] = {
                "centroid": emb.tolist(),
                "count": 0,
                "palette": [],
                "voice_tokens": {},
                "expression_counts": {},
                "pose_prior": {},
                "last_seen_page": int(page_index),
                "last_updated_at": "",
            }
            best_sim = 1.0
            created = True

        entry = chars[best_id]
        prev_count = int(entry.get("count", 0))
        centroid = np.array(entry.get("centroid", emb.tolist()), dtype=np.float32)
        blended = (centroid * prev_count + emb) / max(1.0, prev_count + 1.0)
        blended = blended / (np.linalg.norm(blended) + 1e-9)

        palette_now = self._extract_palette(face)
        entry["centroid"] = blended.tolist()
        entry["count"] = prev_count + 1
        entry["palette"] = self._merge_palette(list(entry.get("palette", [])), palette_now)
        entry["voice_tokens"] = self._merge_voice_tokens(
            dict(entry.get("voice_tokens", {})),
            self._voice_tokens(speaker_note),
        )
        expression = self._expression_hint(face)
        expr_counts = dict(entry.get("expression_counts", {}))
        expr_counts[expression] = int(expr_counts.get(expression, 0)) + 1
        entry["expression_counts"] = expr_counts
        entry["pose_prior"] = self._update_pose_prior(
            existing=dict(entry.get("pose_prior", {})),
            update=self._pose_prior(bbox, image_size=image_size),
        )
        entry["last_seen_page"] = int(page_index)
        entry["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        series["seen_faces_total"] = int(series.get("seen_faces_total", 0)) + 1
        series["updated_at"] = datetime.now(timezone.utc).isoformat()
        return best_id, float(best_sim), created

    def update_from_panel(
        self,
        series_id: str,
        image: Image.Image,
        character_db: CharacterDatabase,
        speaker_notes: Optional[Dict[str, str]] = None,
        page_index: int = 0,
    ) -> dict:
        notes = dict(speaker_notes or {})
        faces = character_db.detect_faces(image)
        if not faces:
            return {
                "series_id": series_id,
                "faces_detected": 0,
                "characters_total": len(self._series(series_id).get("characters", {})),
                "new_characters": 0,
                "matches": 0,
            }

        new_chars = 0
        matches = 0
        with self._lock:
            for idx, (crop, bbox) in enumerate(faces):
                note = notes.get(str(idx), "")
                char_id, sim, created = self.upsert_face(
                    series_id=series_id,
                    face=crop,
                    bbox=bbox,
                    image_size=image.size,
                    character_db=character_db,
                    speaker_note=note,
                    page_index=page_index,
                )
                if created:
                    new_chars += 1
                if sim >= self.similarity_threshold:
                    matches += 1
                # Keep deterministic read/update path.
                _ = char_id
            self._save()

        series = self._series(series_id)
        return {
            "series_id": series_id,
            "faces_detected": len(faces),
            "characters_total": len(series.get("characters", {})),
            "new_characters": new_chars,
            "matches": matches,
        }

    def series_palette(self, series_id: str, max_colors: int = 12) -> List[str]:
        series = self._series(series_id)
        palette: List[str] = []
        for data in series.get("characters", {}).values():
            palette.extend(list(data.get("palette", [])))
        deduped: List[str] = []
        seen = set()
        for color in palette:
            if color in seen:
                continue
            seen.add(color)
            deduped.append(color)
        return deduped[:max_colors]

    def series_voice_profile(self, series_id: str) -> Dict[str, str]:
        series = self._series(series_id)
        out: Dict[str, str] = {}
        for char_id, data in series.get("characters", {}).items():
            tokens = dict(data.get("voice_tokens", {}))
            ranked = sorted(tokens.items(), key=lambda kv: kv[1], reverse=True)[:5]
            if ranked:
                out[char_id] = ", ".join(token for token, _ in ranked)
        return out

    def summary(self, series_id: str) -> dict:
        series = self._series(series_id)
        chars = series.get("characters", {})
        return {
            "series_id": series_id,
            "characters_total": len(chars),
            "seen_faces_total": int(series.get("seen_faces_total", 0)),
            "palette_colors": self.series_palette(series_id),
            "voice_profile": self.series_voice_profile(series_id),
            "updated_at": series.get("updated_at", ""),
        }

    def all_series(self) -> Dict[str, dict]:
        return {series_id: self.summary(series_id) for series_id in self._memory.keys()}
