from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from .model_manager import ModelManager
from .utils import clamp_bbox


_JA_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_KO_RE = re.compile(r"[\uac00-\ud7af]")


@dataclass
class OCRRegion:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    lang: str
    backend: str = "unknown"

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "bbox": [int(x), int(y), int(w), int(h)],
            "text": self.text,
            "confidence": float(self.confidence),
            "lang": self.lang,
            "ocr_backend": self.backend,
        }


def detect_speech_regions(image_np: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    regions: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / rh if rh else 0
        area_ratio = area / float(w * h)
        if 0.2 <= aspect <= 5.5 and 0.0008 <= area_ratio <= 0.35:
            regions.append((x, y, rw, rh))

    if not regions:
        regions.append((0, 0, w, h))
    return sorted(regions, key=lambda b: (b[1], b[0]))


def _infer_lang(text: str) -> str:
    if _KO_RE.search(text):
        return "ko"
    if _JA_RE.search(text):
        return "ja"
    return "unknown"


def _run_manga_ocr(manga_ocr_model, crop: Image.Image) -> str:
    try:
        return str(manga_ocr_model(crop)).strip()
    except Exception:
        return ""


def _run_easyocr(easyocr_reader, crop_np: np.ndarray) -> List[tuple]:
    try:
        return easyocr_reader.readtext(crop_np, detail=1, paragraph=False)
    except Exception:
        return []


def _run_tesseract(crop: Image.Image) -> List[dict]:
    try:
        import pytesseract

        data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT)
        out = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            conf = float(data["conf"][i]) if data["conf"][i] not in ("", "-1") else -1
            if text and conf > 20:
                bbox = [
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                ]
                out.append(
                    {
                        "bbox": bbox,
                        "text": text,
                        "confidence": conf / 100.0,
                    }
                )
        return out
    except Exception:
        return []


def run_ocr(image: Image.Image, model_manager: ModelManager, source_lang: str = "auto") -> List[dict]:
    regions, _ = run_ocr_with_report(image=image, model_manager=model_manager, source_lang=source_lang)
    return regions


def run_ocr_with_report(
    image: Image.Image,
    model_manager: ModelManager,
    source_lang: str = "auto",
) -> tuple[List[dict], dict]:
    image = image.convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    regions = detect_speech_regions(image_np)
    manga_ocr = model_manager.load_manga_ocr()
    easyocr_reader = model_manager.load_easyocr()
    attempted = []
    if source_lang in ("ja", "auto"):
        attempted.append("manga_ocr")
    if source_lang in ("ko", "auto"):
        attempted.append("easyocr")
    attempted.append("tesseract")

    results: List[OCRRegion] = []
    for x, y, rw, rh in regions:
        x, y, rw, rh = clamp_bbox(x, y, rw, rh, w, h)
        crop = image.crop((x, y, x + rw, y + rh))
        crop_np = np.array(crop)
        if source_lang in ("ja", "auto") and manga_ocr is not None:
            text = _run_manga_ocr(manga_ocr, crop)
            if text:
                results.append(OCRRegion((x, y, rw, rh), text, 0.92, "ja", "manga_ocr"))
                continue

        if source_lang in ("ko", "auto") and easyocr_reader is not None:
            bubble_hits = 0
            for (pts, text, conf) in _run_easyocr(easyocr_reader, crop_np):
                text = str(text).strip()
                if not text:
                    continue
                min_x = int(min(p[0] for p in pts))
                min_y = int(min(p[1] for p in pts))
                max_x = int(max(p[0] for p in pts))
                max_y = int(max(p[1] for p in pts))
                bx, by, bw, bh = clamp_bbox(x + min_x, y + min_y, max_x - min_x, max_y - min_y, w, h)
                results.append(OCRRegion((bx, by, bw, bh), text, float(conf), _infer_lang(text)))
                results[-1].backend = "easyocr"
                bubble_hits += 1
            if bubble_hits > 0:
                continue

        for item in _run_tesseract(crop):
            bx, by, bw, bh = item["bbox"]
            bbox = clamp_bbox(x + bx, y + by, bw, bh, w, h)
            text = item["text"]
            results.append(
                OCRRegion(
                    bbox=bbox,
                    text=text,
                    confidence=float(item["confidence"]),
                    lang=_infer_lang(text),
                    backend="tesseract",
                )
            )

    dedup = _dedupe_regions(results)
    logger.info("OCR extracted {} regions", len(dedup))
    backend_counts: dict[str, int] = {}
    for region in dedup:
        backend_counts[region.backend] = backend_counts.get(region.backend, 0) + 1
    selected = "none"
    if backend_counts:
        selected = sorted(backend_counts.items(), key=lambda x: x[1], reverse=True)[0][0]
    report = {
        "attempted": attempted,
        "selected": selected,
        "backend_counts": backend_counts,
        "regions": len(dedup),
    }
    return [r.to_dict() for r in dedup], report


def _dedupe_regions(items: List[OCRRegion]) -> List[OCRRegion]:
    out: List[OCRRegion] = []
    seen = set()
    for item in items:
        key = (item.bbox, item.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
