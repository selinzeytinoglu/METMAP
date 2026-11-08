"""Unit tests for calculate_ET_metrics.py (visibility-relative dwell / NPDT and none filtering)."""

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from calculate_ET_metrics import (
    calculate_fixation_events,
    calculate_summary_scores,
    calculate_visibility_relative_dwell,
    compute_visible_frames_from_masks,
)

FPS = 30.0
FRAME_MS = 1000.0 / FPS


def test_npdt_basic_ratio():
    assert calculate_visibility_relative_dwell(500.0, 1000.0) == 0.5
    assert calculate_visibility_relative_dwell(1000.0, 1000.0) == 1.0


def test_npdt_zero_visibility_is_zero():
    assert calculate_visibility_relative_dwell(500.0, 0.0) == 0.0


def test_summary_scores_adds_npdt_column():
    dominant = ["Parent"] * 30 + ["Child"] * 20 + ["Parent"] * 40
    events = calculate_fixation_events(dominant, fps=FPS)

    parent_dwell_ms = 70 * FRAME_MS
    child_dwell_ms = 20 * FRAME_MS
    aoi_visible_ms = {
        "Parent": 100 * FRAME_MS,  # NPDT = 70/100 = 0.70
        "Child": 50 * FRAME_MS,    # NPDT = 20/50  = 0.40
    }

    rows = calculate_summary_scores(events, aoi_visible_ms=aoi_visible_ms)
    by_aoi = {r["aoi_name"]: r for r in rows}

    assert set(by_aoi) == {"Parent", "Child"}
    assert math.isclose(by_aoi["Parent"]["dwell_time"], parent_dwell_ms, rel_tol=1e-9)
    assert math.isclose(by_aoi["Child"]["dwell_time"], child_dwell_ms, rel_tol=1e-9)
    assert math.isclose(by_aoi["Parent"]["visibility_relative_dwell"], 0.70, rel_tol=1e-9)
    assert math.isclose(by_aoi["Child"]["visibility_relative_dwell"], 0.40, rel_tol=1e-9)


def test_summary_scores_absent_visibility_yields_zero():
    dominant = ["Parent"] * 10 + ["Child"] * 10
    events = calculate_fixation_events(dominant, fps=FPS)

    rows = calculate_summary_scores(events, aoi_visible_ms={"Parent": 100 * FRAME_MS})
    by_aoi = {r["aoi_name"]: r for r in rows}

    assert math.isclose(by_aoi["Parent"]["visibility_relative_dwell"], 0.10, rel_tol=1e-9)
    assert by_aoi["Child"]["visibility_relative_dwell"] == 0.0


def test_backward_compatible_without_visibility_map():
    dominant = ["Parent"] * 5 + ["Child"] * 5
    events = calculate_fixation_events(dominant, fps=FPS)

    rows = calculate_summary_scores(events)
    assert rows
    assert all("visibility_relative_dwell" not in r for r in rows)


def test_trackloss_none_is_filtered():
    dominant = ["none"] * 10 + ["Parent"] * 15 + ["None"] * 5 + ["Child"] * 20 + ["none"] * 8
    events = calculate_fixation_events(dominant, fps=FPS, include_none=False)
    aoi_names = [e["aoi_name"] for e in events]
    assert aoi_names == ["Parent", "Child"], f"Expected ['Parent', 'Child'], got {aoi_names}"

    events_with_none = calculate_fixation_events(dominant, fps=FPS, include_none=True)
    assert len(events_with_none) == 5


def test_compute_visible_frames_from_masks():
    with tempfile.TemporaryDirectory() as tmpdir:
        masks_dir = Path(tmpdir)
        m_present = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        m_absent = np.zeros((2, 2), dtype=np.uint8)

        np.savez_compressed(masks_dir / "frame_00000.npz", Parent=m_present, Child=m_present)
        np.savez_compressed(masks_dir / "frame_00001.npz", Parent=m_present, Child=m_absent)
        np.savez_compressed(masks_dir / "frame_00002.npz", Parent=m_absent, Child=m_absent)

        counts = compute_visible_frames_from_masks(masks_dir)
        assert counts == {"Parent": 2, "Child": 1}


if __name__ == "__main__":
    tests = [
        test_npdt_basic_ratio,
        test_npdt_zero_visibility_is_zero,
        test_summary_scores_adds_npdt_column,
        test_summary_scores_absent_visibility_yields_zero,
        test_backward_compatible_without_visibility_map,
        test_trackloss_none_is_filtered,
        test_compute_visible_frames_from_masks,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")

