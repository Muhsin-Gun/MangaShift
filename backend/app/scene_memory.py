from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Settings


@dataclass
class ScenePageState:
    page_index: int
    style_id: str
    translated_lines: List[str] = field(default_factory=list)
    source_lines: List[str] = field(default_factory=list)
    mood_hint: str = ""
    lighting_hint: str = ""
    palette_anchor: List[str] = field(default_factory=list)
    speaker_notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChapterState:
    pages: Dict[int, ScenePageState] = field(default_factory=dict)
    recent_indices: List[int] = field(default_factory=list)


class SceneMemory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._series: Dict[str, Dict[str, ChapterState]] = {}

    def _chapter_state(self, series_id: str, chapter_id: str) -> ChapterState:
        series_key = series_id or "default_series"
        chapter_key = chapter_id or "default_chapter"
        self._series.setdefault(series_key, {})
        self._series[series_key].setdefault(chapter_key, ChapterState())
        return self._series[series_key][chapter_key]

    def update_page(
        self,
        series_id: str,
        chapter_id: str,
        page_index: int,
        style_id: str,
        source_lines: List[str],
        translated_lines: List[str],
        mood_hint: str = "",
        lighting_hint: str = "",
        palette_anchor: Optional[List[str]] = None,
        speaker_notes: Optional[Dict[str, str]] = None,
    ) -> None:
        chapter = self._chapter_state(series_id, chapter_id)
        state = ScenePageState(
            page_index=int(page_index),
            style_id=style_id,
            source_lines=[s.strip() for s in source_lines if s.strip()],
            translated_lines=[s.strip() for s in translated_lines if s.strip()],
            mood_hint=mood_hint.strip(),
            lighting_hint=lighting_hint.strip(),
            palette_anchor=list(palette_anchor or []),
            speaker_notes=dict(speaker_notes or {}),
        )
        chapter.pages[int(page_index)] = state
        if int(page_index) in chapter.recent_indices:
            chapter.recent_indices.remove(int(page_index))
        chapter.recent_indices.append(int(page_index))
        max_pages = max(1, self.settings.translation_context_window_pages * 3)
        while len(chapter.recent_indices) > max_pages:
            stale_index = chapter.recent_indices.pop(0)
            chapter.pages.pop(stale_index, None)

    def translation_context(
        self,
        series_id: str,
        chapter_id: str,
        page_index: int,
        current_lines: List[str],
    ) -> str:
        chapter = self._chapter_state(series_id, chapter_id)
        page_index = int(page_index)
        previous = [i for i in chapter.recent_indices if i < page_index]
        previous = previous[-self.settings.translation_context_window_pages:]
        context_lines: List[str] = []
        for idx in previous:
            state = chapter.pages.get(idx)
            if not state:
                continue
            if state.source_lines:
                context_lines.extend(state.source_lines[-6:])
            if state.translated_lines:
                context_lines.extend(state.translated_lines[-6:])
        context_lines.extend([line for line in current_lines if line.strip()])
        context_lines = context_lines[-self.settings.translation_context_window_lines:]
        return " | ".join(context_lines)

    def scene_style_hint(self, series_id: str, chapter_id: str, page_index: int) -> Optional[str]:
        chapter = self._chapter_state(series_id, chapter_id)
        previous = [i for i in chapter.recent_indices if i < int(page_index)]
        if not previous:
            return None
        last = chapter.pages.get(previous[-1])
        if not last:
            return None
        return last.style_id

    def palette_anchor(self, series_id: str, chapter_id: str, page_index: int) -> List[str]:
        chapter = self._chapter_state(series_id, chapter_id)
        previous = [i for i in chapter.recent_indices if i <= int(page_index)]
        previous = previous[-self.settings.translation_context_window_pages:]
        palette: List[str] = []
        for idx in previous:
            state = chapter.pages.get(idx)
            if not state:
                continue
            palette.extend(state.palette_anchor)
        # Keep order while deduplicating.
        seen = set()
        deduped = []
        for color in palette:
            if color in seen:
                continue
            seen.add(color)
            deduped.append(color)
        return deduped[:12]
