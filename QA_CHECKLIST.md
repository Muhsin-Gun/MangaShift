# QA Checklist

## 1) Functional Validation

1. Start backend and confirm `GET /health` returns `status=ok`.
2. Load extension unpacked, open popup, verify `Connected` state.
3. Open a manga/manhwa reader page and verify panel replacement while scrolling.
4. Confirm retry overlay appears when backend is offline.
5. Confirm per-series style memory:
   - Enable "Remember style for this series"
   - Select style
   - Reload chapter and verify style auto-applies

## 2) Translation Quality Gate

1. Build a 100-bubble validation set (KO/JA with human references).
2. Measure:
   - adequacy score (1-5)
   - fluency score (1-5)
   - honorific handling correctness
3. Target:
   - average adequacy >= 4.0
   - average fluency >= 4.0
   - honorific consistency >= 90%
4. Review idioms and SFX separately (manual QA pass required).

## 3) Visual Quality Gate

1. Inpaint check: verify removed text regions have no visible ghosting.
2. Typesetting check: translated text fits inside bubbles with no clipping.
3. Upscaling check: line art remains sharp, no heavy halo artifacts.
4. Style check: selected preset changes are visible and stable across chapter pages.
5. Colorization check: same character keeps consistent tone/hair color page-to-page.

## 4) Performance Targets

1. CPU fallback mode:
   - translation-only page < 5s
   - full enhancement page < 25s
2. GPU target mode (12GB+ VRAM):
   - translation-only page < 2s
   - full enhancement page < 12s
3. Cache target:
   - repeated panel response < 500ms

## 5) Character Consistency Validation

1. Run embedding extraction on a chapter sample.
2. Validate cluster purity manually.
3. For recurring characters:
   - similarity score to cluster centroid stays above threshold
4. If using LoRA:
   - compare generated character face crops across pages
   - ensure no major drift in key features

## 6) Failure/Recovery Tests

1. Kill backend during panel processing; extension must show retry state.
2. Restart backend and retry from overlay button; panel should recover.
3. Corrupt model path; backend should return clear error JSON, not crash process.
4. Remove GPU access; pipeline should continue with CPU fallback.

## 7) Logging & Debug Capture

1. Backend logs include per-step timings (`ocr`, `translate`, `inpaint`, `style`, etc.).
2. Extension console logs processing failures with source URL.
3. Save failed panel source and output for triage.

## 8) Release Readiness

1. `pytest tests -q` passes.
2. CI lint + tests pass.
3. Docker image builds successfully.
4. Privacy/legal docs included and reviewed.
