from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .model_manager import ModelManager

_KO_RE = re.compile(r"[\uac00-\ud7af]")
_JA_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

_SERIES_TRANSLATION_MEMORY: Dict[str, Dict[str, str]] = {}
_SERIES_MEMORY_LIMIT = 5000

HONORIFIC_HINTS = {
    "\uc624\ube60": "oppa",
    "\ud615": "hyung",
    "\ub204\ub098": "noona",
    "\uc5b8\ub2c8": "unnie",
    "\uc120\ubc30": "sunbae",
    "\u3061\u3083\u3093": "chan",
    "\u304f\u3093": "kun",
    "\u5148\u8f29": "senpai",
    "\u69d8": "sama",
}

IDIOM_HINTS = {
    "\ub208\uc5d0 \ub123\uc5b4\ub3c4 \uc548 \uc544\ud504\ub2e4": "I cherish them more than anything.",
    "\u3057\u3087\u3046\u304c\u306a\u3044": "It can't be helped.",
    "\u4ed5\u65b9\u306a\u3044": "It can't be helped.",
}


def detect_text_lang(text: str) -> str:
    if _KO_RE.search(text):
        return "ko"
    if _JA_RE.search(text):
        return "ja"
    return "unknown"


def _normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _memory_put(series_id: str, source: str, translated: str) -> None:
    bucket = _SERIES_TRANSLATION_MEMORY.setdefault(series_id, {})
    bucket[source] = translated
    if len(bucket) > _SERIES_MEMORY_LIMIT:
        keep_keys = list(bucket.keys())[-(_SERIES_MEMORY_LIMIT // 2):]
        _SERIES_TRANSLATION_MEMORY[series_id] = {k: bucket[k] for k in keep_keys}


def _translate_group(model_manager: ModelManager, texts: List[str], lang: str) -> Tuple[List[str], str]:
    if not texts:
        return [], "identity"
    bundle = model_manager.load_ko_en() if lang == "ko" else model_manager.load_ja_en()
    if bundle is not None:
        return model_manager.translate_with_bundle(bundle, texts), "marian"

    tok, model = model_manager.load_m2m100()
    if tok is not None and model is not None and model_manager.torch is not None:
        try:
            tok.src_lang = lang
            encoded = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(model_manager.device) for k, v in encoded.items()}
            forced_bos = tok.get_lang_id("en")
            with model_manager.torch.no_grad():
                generated = model.generate(**encoded, forced_bos_token_id=forced_bos, max_new_tokens=256)
            return [tok.decode(g, skip_special_tokens=True) for g in generated], "m2m100"
        except Exception as exc:
            logger.warning("M2M fallback translation failed for {}: {}", lang, exc)
    return texts, "identity"


def _apply_honorific_logic(original: str, translated: str, speaker_note: str = "") -> str:
    out = translated
    for token, hint in HONORIFIC_HINTS.items():
        if token in original and hint.lower() not in out.lower():
            out = f"{out} ({hint})"
    if speaker_note:
        note = speaker_note.lower()
        if "formal" in note and not out.lower().startswith(("sir", "ma'am", "mr.", "ms.")):
            out = f"Sir, {out}" if "male" in note else f"{out}"
    return out


def _apply_tone(original: str, translated: str) -> str:
    out = translated.strip()
    if not out:
        return out
    exclamations = original.count("!")
    questions = original.count("?")
    ellipsis = "..." in original or "\u2026" in original
    if questions > 0 and "?" not in out:
        out = out.rstrip(".!") + "?"
    if exclamations >= 2 and not out.endswith("!!"):
        out = out.rstrip(".!?") + "!!"
    elif exclamations == 1 and not out.endswith(("!", "!!")):
        out = out.rstrip(".?") + "!"
    if ellipsis and "..." not in out:
        out = out.rstrip(".!?") + "..."
    return out


def _idiom_fix(original: str, translated: str) -> str:
    for key, value in IDIOM_HINTS.items():
        if key in original:
            return value
    return translated


def _llm_post_edit_if_available(
    model_manager: ModelManager,
    originals: List[str],
    translated: List[str],
    context: str,
    speaker_map: Optional[Dict[int, str]] = None,
) -> Tuple[List[str], bool]:
    llm = model_manager.load_llm()
    if llm is None:
        return translated, False
    out: List[str] = []
    for idx, (src, raw) in enumerate(zip(originals, translated)):
        speaker_note = speaker_map.get(idx, "") if speaker_map else ""
        prompt = (
            "You are a professional manga localizer.\n"
            f"Original line: {src}\n"
            f"Raw translation: {raw}\n"
            f"Page context: {context}\n"
            f"Speaker notes: {speaker_note}\n"
            "Rules:\n"
            "- Keep meaning and tone.\n"
            "- Preserve emotional intensity.\n"
            "- Preserve honorific intent.\n"
            "- Output one natural English line only."
        )
        try:
            response = llm(prompt, max_tokens=128, stop=["\n\n"], echo=False)
            candidate = response["choices"][0]["text"].strip()
            out.append(candidate or raw)
        except Exception:
            out.append(raw)
    return out, True


def translate_regions(
    regions: List[dict],
    model_manager: ModelManager,
    context: str = "",
    series_id: str = "default_series",
    chapter_id: str = "default_chapter",
    page_index: int = 0,
    speaker_map: Optional[Dict[int, str]] = None,
    relationship_map: Optional[Dict[str, str]] = None,
    use_series_memory: bool = True,
    return_report: bool = False,
) -> List[dict] | Tuple[List[dict], dict]:
    if not regions:
        if return_report:
            return [], {
                "selected": "skipped",
                "backend_counts": {},
                "llm_post_edit": False,
                "series_memory_hits": 0,
                "regions": 0,
            }
        return []
    del chapter_id, page_index, relationship_map

    texts = [_normalize_spaces(str(r.get("text", ""))) for r in regions]
    langs = [detect_text_lang(text) for text in texts]
    ko_idx = [i for i, lang in enumerate(langs) if lang == "ko"]
    ja_idx = [i for i, lang in enumerate(langs) if lang == "ja"]
    unk_idx = [i for i, lang in enumerate(langs) if lang == "unknown"]
    backend_counts: Dict[str, int] = {}
    memory_hits = 0

    translated: Dict[int, str] = {}
    if ko_idx:
        ko_texts = [texts[i] for i in ko_idx]
        ko_out, ko_backend = _translate_group(model_manager, ko_texts, "ko")
        backend_counts[ko_backend] = backend_counts.get(ko_backend, 0) + len(ko_out)
        for idx, value in zip(ko_idx, ko_out):
            translated[idx] = value
    if ja_idx:
        ja_texts = [texts[i] for i in ja_idx]
        ja_out, ja_backend = _translate_group(model_manager, ja_texts, "ja")
        backend_counts[ja_backend] = backend_counts.get(ja_backend, 0) + len(ja_out)
        for idx, value in zip(ja_idx, ja_out):
            translated[idx] = value
    for idx in unk_idx:
        translated[idx] = texts[idx]
        backend_counts["identity"] = backend_counts.get("identity", 0) + 1

    ordered = [translated.get(i, texts[i]) for i in range(len(texts))]
    ordered = [_idiom_fix(src, out) for src, out in zip(texts, ordered)]

    if use_series_memory:
        memory_bucket = _SERIES_TRANSLATION_MEMORY.setdefault(series_id, {})
        for idx, src in enumerate(texts):
            cached = memory_bucket.get(src)
            if cached:
                ordered[idx] = cached
                memory_hits += 1

    ordered, llm_used = _llm_post_edit_if_available(
        model_manager=model_manager,
        originals=texts,
        translated=ordered,
        context=context,
        speaker_map=speaker_map,
    )

    finalized: List[str] = []
    for idx, (src, out) in enumerate(zip(texts, ordered)):
        speaker_note = speaker_map.get(idx, "") if speaker_map else ""
        line = _apply_honorific_logic(src, out, speaker_note=speaker_note)
        line = _apply_tone(src, line)
        finalized.append(line)
        if use_series_memory and src:
            _memory_put(series_id, src, line)

    result = []
    for idx, region in enumerate(regions):
        item = dict(region)
        item["translated_text"] = finalized[idx]
        item["translation_lang"] = langs[idx]
        result.append(item)

    logger.info("Translated {} regions (series={})", len(result), series_id)
    if not return_report:
        return result

    selected_backend = "identity"
    if backend_counts:
        selected_backend = sorted(backend_counts.items(), key=lambda x: x[1], reverse=True)[0][0]
    report = {
        "selected": selected_backend,
        "backend_counts": backend_counts,
        "llm_post_edit": llm_used,
        "series_memory_hits": memory_hits,
        "regions": len(result),
    }
    return result, report
