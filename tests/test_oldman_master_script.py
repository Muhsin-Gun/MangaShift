from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "backend" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_oldman_master as oldman  # noqa: E402


def test_parse_crop_valid():
    crop = oldman.parse_crop("460,120,1260,1110")
    assert crop == (460, 120, 1260, 1110)


def test_parse_crop_invalid():
    try:
        oldman.parse_crop("10,10,10,20")
        assert False, "expected ValueError for invalid crop"
    except ValueError:
        pass
