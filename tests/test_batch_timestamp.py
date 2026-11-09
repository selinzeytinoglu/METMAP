"""Unit tests for metadata-CSV frame and timestamp resolution in batch_processor.py."""

import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from batch_processor import resolve_frame_range

FPS = 29.96


def test_frame_columns():
    row = {"Participant_ID": 510, "Start_Frame": 3997, "End_Frame": 9393}
    assert resolve_frame_range(row, FPS) == (3997, 9393)
    print("PASS: frame columns")


def test_seconds_columns():
    row = {"Participant_ID": 510, "Start_Sec": 133.4, "End_Sec": 313.5}
    assert resolve_frame_range(row, FPS) == (int(133.4 * FPS), int(313.5 * FPS))
    print("PASS: seconds columns converted via fps")


def test_frames_take_precedence():
    row = {"Start_Frame": 10, "End_Frame": 20, "Start_Sec": 1.0, "End_Sec": 2.0}
    assert resolve_frame_range(row, FPS) == (10, 20)
    print("PASS: frame columns win when both present")


def test_time_alias_columns():
    row = {"Start_Time": 2.0, "End_Time": 4.0}
    assert resolve_frame_range(row, FPS) == (int(2.0 * FPS), int(4.0 * FPS))
    print("PASS: Start_Time/End_Time alias")


def test_missing_raises():
    try:
        resolve_frame_range({"Participant_ID": 510}, FPS)
    except ValueError:
        print("PASS: raises when neither pair present")
        return
    raise AssertionError("expected ValueError")


def test_nan_and_blank_handled():
    row = {"Start_Frame": float("nan"), "End_Frame": "", "Start_Sec": 1.0, "End_Sec": 2.0}
    assert resolve_frame_range(row, FPS) == (int(1.0 * FPS), int(2.0 * FPS))
    print("PASS: NaN/blank falls through to time columns")


def test_hardware_timestamps_array_handles_dropped_frames():
    # 30 fps nominal, starting at t=0.0, with 2 dropped frames after index 10 (100ms jump)
    ts = []
    t = 0.0
    for i in range(100):
        ts.append(t)
        t += 0.100 if i == 10 else 1.0 / 30.0

    row = {"Start_Sec": 1.0, "End_Sec": 2.0}
    s, e = resolve_frame_range(row, FPS, timestamps=ts)
    assert s == 28, f"Expected frame 28, got {s}"
    print("PASS: hardware timestamps array resolves dropped frames")


def test_hardware_timestamps_file_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "world_timestamps.csv"
        # Non-zero monotonic clock start (e.g., 2.77s at frame 0, as in Pupil Invisible exports)
        csv_file.write_text("world_timestamp\n2.7700\n2.8033\n2.8367\n2.9367\n2.9700\n", encoding="utf-8")

        row = {"Start_Sec": 0.150, "End_Sec": 0.200, "Timestamps_File": str(csv_file)}
        s, e = resolve_frame_range(row, FPS)
        assert (s, e) == (3, 4)
        print("PASS: hardware timestamps CSV column resolved via searchsorted (with non-zero boot offset)")


if __name__ == "__main__":
    test_frame_columns()
    test_seconds_columns()
    test_frames_take_precedence()
    test_time_alias_columns()
    test_missing_raises()
    test_nan_and_blank_handled()
    test_hardware_timestamps_array_handles_dropped_frames()
    test_hardware_timestamps_file_column()
    print("All batch timestamp resolution tests passed.")
