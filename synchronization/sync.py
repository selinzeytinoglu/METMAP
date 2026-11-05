# sync.py — Video synchronization utilities for multi-device eye-tracking research.
#
# Uses a lights-on/lights-off protocol: the experimenter dims the room lights before
# a task, creating a luminance event visible across all simultaneous recordings.
# Each video's lights-on frame serves as the synchronization marker.
#
# Workflow:
#   1. find_lights_on() — locate the sync marker in each video
#   2. align_timestamps() — translate event times between video timelines
#   3. timestamps_to_frames() — convert aligned timestamps to frame numbers
#
# See run_sync.py for a batch example.

import cv2
import numpy as np
from typing import List, Optional, Tuple


def find_lights_on(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    brightness_threshold: float = 50.0,
) -> Tuple[Optional[float], Optional[int]]:
    # Returns (timestamp_sec, frame_number) of the first dark-to-bright transition,
    # or (None, None) if the event is not detected in the scan window.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec is not None else float("inf")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    in_dark = False

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not ret or current_frame > end_frame:
                break

            is_dark = _mean_brightness(frame) < brightness_threshold

            if not in_dark and is_dark:
                in_dark = True
            elif in_dark and not is_dark:
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                return timestamp_sec, int(current_frame)
    finally:
        cap.release()

    return None, None


def align_timestamps(
    timestamps_sec: List[float],
    source_lights_on_sec: float,
    target_lights_on_sec: float,
) -> List[float]:
    # Translate event timestamps from the source video's timeline to the target's
    # using the lights-on timestamps as the shared anchor point.
    offset = target_lights_on_sec - source_lights_on_sec
    return [t + offset for t in timestamps_sec]


def timestamps_to_frames(timestamps_sec: List[float], fps: float) -> List[int]:
    return [int(t * fps) for t in timestamps_sec]


def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# ---------------------------------------------------------------------------
# Time-string parsing utilities
# ---------------------------------------------------------------------------

def hms_to_sec(time_str: str) -> float:
    # Accepts HH:MM:SS or MM:SS.
    parts = list(map(int, time_str.strip().split(":")))
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def msm_to_sec(time_str: str) -> float:
    # Accepts MM:SS or MM:SS:mmm (minutes:seconds:milliseconds).
    parts = list(map(int, time_str.strip().split(":")))
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return parts[0] * 60 + parts[1] + parts[2] / 1000.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_brightness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
