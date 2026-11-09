"""Unit tests for glasses_adapter.py (multi-tracker gaze ingestion)."""

import os
import subprocess
import sys
import tempfile

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "sam2_pipeline", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from glasses_adapter import load_universal_gaze


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def test_glassestools_pixel_coords_normalized():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gaze.tsv")
        _write(
            path,
            "frame_idx\tgaze_x\tgaze_y\ttimestamp\n"
            "0\t640\t360\t0.0\n"
            "1\t320\t180\t0.033\n",
        )
        df = load_universal_gaze(
            path,
            video_width=1280,
            video_height=720,
            coordinate_origin="bottom_left",
            timestamp_unit="s",
        )

        assert list(df.columns) == ["frame_idx", "gaze_x_norm", "gaze_y_norm", "timestamp_sec"]
        assert df["frame_idx"].tolist() == [0, 1]
        assert abs(df["gaze_x_norm"].iloc[0] - 0.5) < 1e-9
        assert abs(df["gaze_y_norm"].iloc[0] - 0.5) < 1e-9
        assert abs(df["gaze_x_norm"].iloc[1] - 0.25) < 1e-9


def test_pupil_labs_autodetect():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pupil.csv")
        _write(
            path,
            "world_index,norm_pos_x,norm_pos_y,world_timestamp\n"
            "29,0.42,0.40,2.77\n",
        )
        df = load_universal_gaze(path)

        assert df["frame_idx"].tolist() == [29]
        assert abs(df["gaze_x_norm"].iloc[0] - 0.42) < 1e-9
        assert abs(df["gaze_y_norm"].iloc[0] - 0.40) < 1e-9
        assert abs(df["timestamp_sec"].iloc[0] - 2.77) < 1e-9


def test_tobii_autodetect_flips_top_left_and_scales_ms():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tobii.csv")
        _write(
            path,
            "Frame number,Gaze point 2D X,Gaze point 2D Y,Recording timestamp\n"
            "15,0.25,0.20,1500\n",
        )
        df = load_universal_gaze(path)

        assert df["frame_idx"].tolist() == [15]
        assert abs(df["gaze_x_norm"].iloc[0] - 0.25) < 1e-9
        assert abs(df["gaze_y_norm"].iloc[0] - 0.80) < 1e-9  # 1.0 - 0.20
        assert abs(df["timestamp_sec"].iloc[0] - 1.50) < 1e-9


def test_cli_entry_point_writes_standardized_csv():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "gaze.tsv")
        out = os.path.join(d, "standardized.csv")
        _write(
            src,
            "frame_idx\tgaze_x\tgaze_y\ttimestamp\n"
            "0\t128\t72\t0.0\n",
        )
        script = os.path.join(SCRIPTS_DIR, "glasses_adapter.py")
        result = subprocess.run(
            [
                sys.executable, script, src,
                "--video-width", "1280", "--video-height", "720",
                "--coordinate-origin", "top_left", "--timestamp-unit", "s",
                "--output", out,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert list(df.columns) == ["frame_idx", "gaze_x_norm", "gaze_y_norm", "timestamp_sec"]
        assert abs(df["gaze_x_norm"].iloc[0] - 0.1) < 1e-9
        assert abs(df["gaze_y_norm"].iloc[0] - 0.9) < 1e-9


if __name__ == "__main__":
    for t in (
        test_glassestools_pixel_coords_normalized,
        test_pupil_labs_autodetect,
        test_tobii_autodetect_flips_top_left_and_scales_ms,
        test_cli_entry_point_writes_standardized_csv,
    ):
        t()
        print(f"PASS: {t.__name__}")
    print("All glasses_adapter tests passed.")

