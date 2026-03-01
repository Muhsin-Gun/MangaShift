# Data Guide

## Parallel Corpus for Manga/Manhwa Translation

Use legally obtained pages only. Recommended starting sources:
- Manga109 metadata for research workflows
- Your own licensed/manually curated chapter captures

## Annotation Schema (JSONL)

Each line:
```json
{
  "image_path": "chapters/ch01/page_003.png",
  "bbox": [132, 245, 180, 92],
  "original_text": "원문 또는 日本語",
  "corrected_text": "OCR-corrected source line",
  "translation": "final English line",
  "speaker": "char_kariel",
  "type": "speech"
}
```

`type` values:
- `speech`
- `thought`
- `sfx`
- `narration`

## Minimum Dataset Recommendation

- 5,000+ bubble pairs (KO/JA -> EN)
- 15-20 series minimum for style/voice coverage
- 20% holdout validation split

## Labeling Rules

1. Keep speaker labels stable across chapter.
2. Preserve punctuation and emphasis markers.
3. Mark honorifics explicitly in source and translation notes.
4. Separate SFX from dialogue.
5. Add context ID for multi-bubble conversations in same panel.

## Preprocessing

Suggested commands:
```powershell
# Normalize format to PNG
magick mogrify -format png chapters\**\*.jpg

# Denoise lightly
python - <<'PY'
import cv2, glob
for p in glob.glob('chapters/**/*.png', recursive=True):
    img = cv2.imread(p)
    out = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    cv2.imwrite(p, out)
PY
```

## OCR QA Loop

1. OCR raw extraction
2. Human correction UI (or spreadsheet)
3. Save corrected source text for MT training/eval

## Evaluation Metrics

- COMET/BLEU for baseline objective comparison
- human adequacy + fluency for final quality decision
- terminology consistency rate by character
