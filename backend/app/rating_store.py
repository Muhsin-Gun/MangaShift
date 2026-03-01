from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


class RatingStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ratings_path = self.base_dir / "ab_ratings.jsonl"

    def append(self, payload: Dict[str, Any]) -> None:
        row = dict(payload)
        row["timestamp"] = time.time()
        with self.ratings_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not self.ratings_path.exists():
            return []
        lines = self.ratings_path.read_text(encoding="utf-8").splitlines()
        rows: List[Dict[str, Any]] = []
        for raw in lines[-max(1, int(limit)) :]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    def list_runs(self) -> List[Dict[str, Any]]:
        runs_dir = self.base_dir / "variant_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
            metadata_file = run_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
            out.append(
                {
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "best_index": metadata.get("best_index"),
                    "variant_count": metadata.get("variant_count", 0),
                    "style_name": metadata.get("style_name", ""),
                    "render_quality": metadata.get("render_quality", ""),
                }
            )
        return out

    def run_metadata(self, run_id: str) -> Dict[str, Any]:
        run_dir = self.base_dir / "variant_runs" / str(run_id)
        metadata_file = run_dir / "metadata.json"
        if not metadata_file.exists():
            return {}
        try:
            return json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
