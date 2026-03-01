# Privacy Policy (Local-First)

## Core Principle

MangaShift AI runs locally by default. Page images are processed on your machine through `http://127.0.0.1:8000`.

## What Data Is Processed

- Manga/manhwa panel images visible in your browser tab
- OCR-detected text regions for translation
- Local cache entries for faster repeat processing
- User settings (style/language/toggles) in Chrome extension storage

## What Is Not Sent by Default

- No automatic upload to external cloud services
- No analytics telemetry by default
- No remote storage of your pages by default

## Local Storage

- Extension cache: IndexedDB
- Backend cache: local disk (`backend/cache/`)
- Settings: `chrome.storage.sync`

You can clear extension storage from browser settings and delete backend cache folder manually.

## Legal / Copyright Notice

Manga and manhwa pages are copyrighted works. This tool is intended for personal local transformation and accessibility workflows.

Do not redistribute transformed pages unless you hold rights or license permissions from the copyright owner.

If cloud processing is added later, explicit opt-in, retention policy, and transparency controls are required.
