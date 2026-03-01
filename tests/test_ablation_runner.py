from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "backend" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import run_ablation as ra  # noqa: E402


def test_percentile_midpoint():
    values = [10.0, 20.0, 30.0, 40.0]
    assert ra.percentile(values, 0.5) == 25.0
    assert ra.percentile(values, 0.0) == 10.0
    assert ra.percentile(values, 1.0) == 40.0


def test_summarize_records_and_score():
    records = [
        {
            "status": 200,
            "latency_ms": 120.0,
            "total_ms": 110.0,
            "edge_overlap": 0.8,
            "blur_score": 0.9,
            "contrast_shift": 10.0,
            "texture_shift": 0.05,
            "identity_similarity": 0.88,
            "planner_applied": True,
            "style_backend": "fallback_filter",
            "adaptive_strength": 0.38,
            "adaptive_steps": 14,
        },
        {
            "status": 200,
            "latency_ms": 140.0,
            "total_ms": 130.0,
            "edge_overlap": 0.82,
            "blur_score": 0.95,
            "contrast_shift": 12.0,
            "texture_shift": 0.04,
            "identity_similarity": 0.9,
            "planner_applied": False,
            "style_backend": "fallback_filter",
            "adaptive_strength": 0.36,
            "adaptive_steps": 13,
        },
        {"status": 500, "latency_ms": 50.0},
    ]
    summary = ra.summarize_records(records, quality="balanced")
    assert summary["total_requests"] == 3
    assert summary["success_requests"] == 2
    assert 0.0 <= summary["overall_score"] <= 100.0
    assert "200" in summary["status_counts"]
    assert "fallback_filter" in summary["style_backend_counts"]
