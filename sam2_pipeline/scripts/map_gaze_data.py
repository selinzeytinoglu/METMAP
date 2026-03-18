'''
This program takes
1.) Extracted video frames directory
2.) Start and end frames for the video
3.) gaze_positions.csv
4.) sam2 ROI masks
as inputs and processes the video frame by frame to see if the persons gaze (retrieved from gaze_positions.csv)
intersects with the masks for the ROI's that were generated using sam2.

The output is a fixations.csv which contains where the script determines someone is looking for every frame
and an output video that displays the gaze and masks for visual inspection.
'''

import math
import os
import yaml
import cv2
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np

# Color palette for mask overlays (BGR format for OpenCV)
MASK_COLORS = [
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (0, 0, 255),    # red
    (255, 0, 255),  # magenta
    (0, 255, 0),    # green
    (255, 255, 0),  # cyan
]

def map_gaze_data(base_dir, frames_dir, gaze_path, masks_dir, start_frame, end_frame, id, fps, uncertainty_radius=0):
    out_dir = os.path.join(base_dir, f"{id}")
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, f"{id}_output.avi")

    # Get video dimensions from first frame image
    frame_names = [f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')]
    frame_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
    first_frame_img = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    height, width = first_frame_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # Load gaze data
    frames = pd.read_csv(gaze_path)

    # Setup fixations
    prev_fixation = None
    fixation_duration = 0

    output_csv_path = os.path.join(out_dir, f"{id}_fixations.csv")
    csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["name", "frame duration", "start frame", "end frame", "duration (s)", "start (s)", "end (s)"])

    expanded_csv_path = os.path.join(out_dir, f"{id}_fixations_expanded.csv")
    expanded_csv_file = open(expanded_csv_path, 'w', newline='')
    expanded_writer = csv.writer(expanded_csv_file)
    expanded_writer.writerow(["framenum", "code"])

    pbar = tqdm(total=end_frame-start_frame, desc="Processing frames", unit="frame")

    masks_dict = {}
    mask_files = os.listdir(masks_dir)
    mask_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for i, file in enumerate(mask_files):
        masks_dict[i] = os.path.join(masks_dir, file)

    # Main loop that goes through every frame
    for i in range(end_frame - start_frame):
        pbar.update(1)

        frame_num = i + start_frame
        gaze_data = frames[frames.world_index == frame_num].iloc[:, 3:5].values

        frame = cv2.imread(os.path.join(frames_dir, frame_names[i]))
        mask_path = masks_dict[i]
        add_masks_to_frame(frame, mask_path)

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
                    roi = check_intersection(mask_path, x_eye, y_eye, uncertainty_radius)

        cv2.putText(frame, f"Intersecting: {roi}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_fixation = roi
        if current_fixation == prev_fixation:
            fixation_duration += 1
        else:
            start_fixation_frame = int(frame_num - fixation_duration)
            end_fixation_frame = int(frame_num)
            if fixation_duration != 0:
                writer.writerow([prev_fixation, int(fixation_duration), start_fixation_frame, end_fixation_frame,
                                round(fixation_duration / fps, 2), round(start_fixation_frame / fps, 2), round(end_fixation_frame / fps, 2)])
            fixation_duration = 1
            prev_fixation = current_fixation

        expanded_writer.writerow([i+1, roi])
        out_video.write(frame)

    # Write the last fixation
    start_fixation_frame = int(frame_num - fixation_duration)
    end_fixation_frame = int(frame_num)
    if fixation_duration != 0:
        writer.writerow([prev_fixation, int(fixation_duration), start_fixation_frame, end_fixation_frame,
                        round(fixation_duration / fps, 2), round(start_fixation_frame / fps, 2), round(end_fixation_frame / fps, 2)])

    out_video.release()
    csv_file.close()
    expanded_csv_file.close()


def check_intersection(mask_path, x_eye, y_eye, radius):
    masks_data = np.load(mask_path)

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


def add_masks_to_frame(frame, mask_path):
    masks_data = np.load(mask_path)
    overlay = frame.copy()

    for i, key in enumerate(masks_data.keys()):
        mask = masks_data[key]
        color = MASK_COLORS[i % len(MASK_COLORS)]
        overlay[mask > 0] = color

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def p(relative_path):
        return os.path.join(cfg["local_data_dir"], relative_path)

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
