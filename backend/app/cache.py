from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from .utils import ensure_dir


class FileLRUCache:
    def __init__(self, cache_dir: Path, max_items: int = 512, max_size_bytes: int = 10 * 1024**3):
        self.cache_dir = ensure_dir(cache_dir)
        self.max_items = max_items
        self.max_size_bytes = max_size_bytes
        self._index_path = self.cache_dir / "index.json"
        self._lock = threading.Lock()
        self._index: Dict[str, dict] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, ensure_ascii=False), encoding="utf-8")

    def _entry_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.png"

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            entry = self._index.get(key)
            if not entry:
                return None
            path = self._entry_path(key)
            if not path.exists():
                self._index.pop(key, None)
                self._save_index()
                return None
            entry["last_access"] = time.time()
            self._save_index()
        return path.read_bytes()

    def put(self, key: str, payload: bytes) -> Path:
        path = self._entry_path(key)
        path.write_bytes(payload)
        size = path.stat().st_size
        with self._lock:
            self._index[key] = {"size": size, "last_access": time.time()}
            self._evict_if_needed()
            self._save_index()
        return path

    def _evict_if_needed(self) -> None:
        total_size = sum(v.get("size", 0) for v in self._index.values())
        if len(self._index) <= self.max_items and total_size <= self.max_size_bytes:
            return
        for key, _ in sorted(self._index.items(), key=lambda kv: kv[1].get("last_access", 0)):
            if len(self._index) <= self.max_items and total_size <= self.max_size_bytes:
                break
            path = self._entry_path(key)
            size = self._index.get(key, {}).get("size", 0)
            try:
                if path.exists():
                    path.unlink()
            finally:
                self._index.pop(key, None)
                total_size -= size
