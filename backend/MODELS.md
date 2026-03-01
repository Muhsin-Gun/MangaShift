# Model Inventory

## Translation
- `Helsinki-NLP/opus-mt-ja-en` (Marian JA -> EN)
- `Helsinki-NLP/opus-mt-ko-en` (Marian KO -> EN)
- `facebook/m2m100_418M` (fallback multilingual MT)

## OCR
- `manga-ocr` (pip package using `kha-white/manga-ocr-base`)
- `easyocr` (KO/JA fallback OCR)
- `pytesseract` (CPU fallback OCR)

## Inpainting
- `runwayml/stable-diffusion-inpainting` (diffusers inpainting pipeline)
- OpenCV Telea inpaint fallback (no external model)

## Style Transfer / Colorization
- `runwayml/stable-diffusion-v1-5` (img2img style/color operations)
- Optional ControlNet:
  - `lllyasviel/sd-controlnet-canny`

## Upscaling
- `RealESRGAN_x4plus_anime_6B` (anime super-resolution)
- PIL + unsharp fallback

## Optional LLM Post-Edit
- Any GGUF instruct model (example: Mistral 7B Instruct quantized)
- Configure `LLM_GGUF_PATH` in backend `.env`

## Download Commands

```powershell
cd backend
python scripts\download_models.py --all
```

Or selective:
```powershell
python scripts\download_models.py --only marian_ja_en marian_ko_en sd_base
```

All model downloads are cached locally on your machine.

Generate deterministic manifest (recommended before strict mode):
```powershell
python scripts\model_manifest.py --write --verify --models-dir .\models
```

Run strict smoke gate:
```powershell
python scripts\run_strict_smoke.py --strict-ocr --strict-translation --strict-diffusion --quality final
```
