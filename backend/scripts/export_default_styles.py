from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.style_packages import StylePackageManager  # noqa: E402


def main() -> None:
    settings = get_settings()
    manager = StylePackageManager(settings=settings)
    payload = manager.list_dicts()
    out = settings.style_packages_file
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(payload)} style packages to {out}")


if __name__ == "__main__":
    main()
