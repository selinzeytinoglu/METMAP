"""
Map gaze coordinates to SAM2 ROI masks.

Inputs:
1. Extracted video frames directory
2. Start and end frames for the video
3. gaze_positions.csv
4. SAM2 ROI masks

Outputs:
- fixations.csv with frame ranges for inferred gaze targets
- fixations_expanded.csv with one row per processed frame
- output video with gaze and mask overlays for visual inspection
"""

import csv
import math
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
]

MASK_METADATA_PREFIX = "__"


def _npz_scalar_to_str(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _expected_frame_count(start_frame, end_frame):
    count = int(end_frame) - int(start_frame) + 1
    if count <= 0:
        raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")
    return count


def _parse_relative_frame_index(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.startswith("frame_"):
        stem = stem[len("frame_"):]
    try:
        return int(stem)
    except ValueError as exc:
        raise ValueError(f"Cannot parse frame index from filename: {filename}") from exc


def _indexed_files(folder, suffixes):
    suffixes = {suffix.lower() for suffix in suffixes}
    indexed = {}
    for filename in os.listdir(folder):
        if os.path.splitext(filename)[1].lower() not in suffixes:
            continue
        index = _parse_relative_frame_index(filename)
        if index in indexed:
            raise ValueError(
                f"Duplicate relative frame index {index} in {folder}: "
                f"{indexed[index]} and {filename}"
            )
        indexed[index] = filename
    return indexed


def _validate_aligned_inputs(frames_dir, masks_dir, frame_count):
    frame_files = _indexed_files(frames_dir, {".jpg", ".jpeg"})
    mask_files = _indexed_files(masks_dir, {".npz"})
    expected = set(range(frame_count))

    missing_frames = sorted(expected - set(frame_files))
    missing_masks = sorted(expected - set(mask_files))
    if missing_frames or missing_masks:
        raise ValueError(
            "Frame/mask alignment failed: "
            f"missing_frames={missing_frames[:10]}, missing_masks={missing_masks[:10]}, "
            f"expected_count={frame_count}, frame_files={len(frame_files)}, mask_files={len(mask_files)}"
        )

    return (
        [frame_files[index] for index in range(frame_count)],
        [os.path.join(masks_dir, mask_files[index]) for index in range(frame_count)],
    )


def load_masks(mask_path):
    with np.load(mask_path) as masks_data:
        keys = [key for key in masks_data.files if not key.startswith(MASK_METADATA_PREFIX)]
        storage = "bool"
        if "__mask_storage__" in masks_data.files:
            storage = _npz_scalar_to_str(masks_data["__mask_storage__"])

        if storage == "packbits":
            height = int(masks_data["__height__"])
            width = int(masks_data["__width__"])
            bitorder = _npz_scalar_to_str(masks_data.get("__bitorder__", "little"))
            count = height * width
            return {
                key: np.unpackbits(masks_data[key], bitorder=bitorder, count=count)
                .reshape(height, width)
                .astype(bool)
                for key in keys
            }

        return {key: masks_data[key] for key in keys}


def map_gaze_data(base_dir, frames_dir, gaze_path, masks_dir, start_frame, end_frame, id, fps, uncertainty_radius=0):
    out_dir = os.path.join(base_dir, f"{id}")
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, f"{id}_output.avi")

    frame_count = _expected_frame_count(start_frame, end_frame)
    frame_names, mask_paths = _validate_aligned_inputs(frames_dir, masks_dir, frame_count)

    first_frame_img = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    if first_frame_img is None:
        raise RuntimeError(f"Failed to read first frame: {os.path.join(frames_dir, frame_names[0])}")
    height, width = first_frame_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    if not out_video.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {out_video_path}")

    frames = pd.read_csv(gaze_path)
    prev_fixation = None
    fixation_duration = 0
    fixation_start_frame = None

    output_csv_path = os.path.join(out_dir, f"{id}_fixations.csv")
    expanded_csv_path = os.path.join(out_dir, f"{id}_fixations_expanded.csv")

    try:
        with open(output_csv_path, 'w', newline='') as csv_file, open(expanded_csv_path, 'w', newline='') as expanded_csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["name", "frame duration", "start frame", "end frame", "duration (s)", "start (s)", "end (s)"])

            expanded_writer = csv.writer(expanded_csv_file)
            expanded_writer.writerow(["framenum", "code"])

            with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
                for i in range(frame_count):
                    pbar.update(1)

                    frame_num = i + int(start_frame)
                    gaze_data = frames[frames.world_index == frame_num].iloc[:, 3:5].values

                    frame_path = os.path.join(frames_dir, frame_names[i])
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        raise RuntimeError(f"Failed to read frame: {frame_path}")

                    masks_data = load_masks(mask_paths[i])
                    add_masks_to_frame(frame, masks_data)

                    if len(gaze_data) == 0:
                        roi = "none"
                    else:
                        x_norm, y_norm = gaze_data[0].item(0), gaze_data[0].item(1)
                        if math.isnan(x_norm) or math.isnan(y_norm):
                            roi = "none"
                        else:
                            x_eye = int(x_norm * width)
                            y_eye = int((1 - y_norm) * height)
                            if x_eye < 0 or x_eye >= width or y_eye < 0 or y_eye >= height:
                                roi = "none"
                            else:
                                cv2.circle(frame, (x_eye, y_eye), radius=4, color=(0, 255, 0), thickness=-1)
                                cv2.circle(frame, (x_eye, y_eye), radius=uncertainty_radius, color=(0, 255, 0), thickness=2)
                                roi = check_intersection(masks_data, x_eye, y_eye, uncertainty_radius)

                    cv2.putText(frame, f"Intersecting: {roi}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    current_fixation = roi
                    if prev_fixation is None:
                        prev_fixation = current_fixation
                        fixation_duration = 1
                        fixation_start_frame = frame_num
                    elif current_fixation == prev_fixation:
                        fixation_duration += 1
                    else:
                        end_fixation_frame = frame_num - 1
                        writer.writerow([
                            prev_fixation,
                            int(fixation_duration),
                            int(fixation_start_frame),
                            int(end_fixation_frame),
                            round(fixation_duration / fps, 2),
                            round(fixation_start_frame / fps, 2),
                            round(end_fixation_frame / fps, 2),
                        ])
                        prev_fixation = current_fixation
                        fixation_duration = 1
                        fixation_start_frame = frame_num

                    expanded_writer.writerow([frame_num, roi])
                    out_video.write(frame)

            if prev_fixation is not None:
                end_fixation_frame = int(fixation_start_frame + fixation_duration - 1)
                writer.writerow([
                    prev_fixation,
                    int(fixation_duration),
                    int(fixation_start_frame),
                    int(end_fixation_frame),
                    round(fixation_duration / fps, 2),
                    round(fixation_start_frame / fps, 2),
                    round(end_fixation_frame / fps, 2),
                ])
    finally:
        out_video.release()


def check_intersection(masks_data, x_eye, y_eye, radius):

    for key in masks_data.keys():
        mask = masks_data[key]

        if radius == 0:
            if mask[y_eye, x_eye] > 0:
                return key
            continue

        y_min = max(0, y_eye - radius)
        y_max = min(mask.shape[0], y_eye + radius + 1)
        x_min = max(0, x_eye - radius)
        x_max = min(mask.shape[1], x_eye + radius + 1)

        if y_max <= y_min or x_max <= x_min:
            continue

        region = mask[y_min:y_max, x_min:x_max]

        y_indices, x_indices = np.ogrid[:y_max-y_min, :x_max-x_min]
        dist_from_center = np.sqrt((x_indices - (x_eye - x_min))**2 +
                                   (y_indices - (y_eye - y_min))**2)
        circular_mask = dist_from_center <= radius

        if np.any(region[circular_mask] > 0):
            return key

    return "none"


def add_masks_to_frame(frame, masks_data):
    overlay = frame.copy()

    for i, key in enumerate(masks_data.keys()):
        mask = masks_data[key]
        color = MASK_COLORS[i % len(MASK_COLORS)]
        overlay[mask > 0] = color

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


if __name__ == "__main__":
    config_path = "/config/config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def p(relative_path):
        return os.path.join(cfg["docker_data_dir"], relative_path)

    base_dir = p(cfg["output_dir"])
    frames_dir = p(cfg["frames_dir"])
    gaze_path = p(cfg["gaze_path"])
    masks_dir = p(cfg["masks_dir"])
    start_frame = cfg["start_frame"]
    end_frame = cfg["end_frame"]
    id = cfg["participant_id"]
    fps = cfg["fps"]
    uncertainty_radius = cfg.get("uncertainty_radius", 0)

    map_gaze_data(base_dir, frames_dir, gaze_path, masks_dir, start_frame, end_frame, id, fps, uncertainty_radius)
