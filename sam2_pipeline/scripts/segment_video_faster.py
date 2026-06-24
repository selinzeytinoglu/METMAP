import os
import sys
import yaml
import numpy as np
import torch
import json
import cv2
import queue
import threading
from collections import defaultdict
from PIL import Image
from time import perf_counter

# Import SAM2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import build_sam2_video_predictor

def normalize_frame_index(frame):
    if isinstance(frame, str):
        stem = os.path.splitext(os.path.basename(frame))[0]
        if stem.isdigit():
            return int(stem)

        digits = "".join(ch for ch in reversed(stem) if ch.isdigit())
        if digits:
            return int(digits[::-1])

        raise ValueError(f"Cannot parse frame index from '{frame}'")

    return int(frame)

def _get_box_from_dict(box_dict):
    key_sets = [
        ("x1", "y1", "x2", "y2"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]
    for keys in key_sets:
        if all(k in box_dict for k in keys):
            return [box_dict[k] for k in keys]
    raise ValueError(f"Box dict must use one of these key sets: {key_sets}")

def parse_box_prompt(prompt):
    if "box" in prompt:
        box = prompt["box"]
    elif "bbox" in prompt:
        box = prompt["bbox"]
    else:
        box = _get_box_from_dict(prompt)

    if isinstance(box, dict):
        raw_box = _get_box_from_dict(box)
    else:
        raw_box = np.asarray(box, dtype=np.float32).reshape(-1)
        if raw_box.size != 4:
            raise ValueError(f"Box prompt must have four coordinates; got {box}")

    x1, y1, x2, y2 = [float(v) for v in raw_box]
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        raise ValueError(f"Box prompt contains a non-finite coordinate: {box}")

    return np.array(
        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        dtype=np.float32,
    )

def is_box_prompt(prompt):
    box_keys = {"box", "bbox", "x1", "x_min", "xmin", "left"}
    return any(key in prompt for key in box_keys)

def build_prompt_groups(coords):
    prompt_groups = defaultdict(lambda: {"points": [], "labels": [], "boxes": []})

    for label, prompts in coords.items():
        for prompt in prompts:
            frame_idx = normalize_frame_index(prompt.get("frame", 0))
            group = prompt_groups[(label, frame_idx)]

            if is_box_prompt(prompt):
                group["boxes"].append(parse_box_prompt(prompt))
                continue

            if "x" not in prompt or "y" not in prompt:
                raise ValueError(f"Point prompt for label '{label}' frame {frame_idx} is missing x/y: {prompt}")

            group["points"].append([float(prompt["x"]), float(prompt["y"])])
            group["labels"].append(int(prompt.get("positive", 1)))

    return prompt_groups

def print_prompt_audit(prompt_groups, label_order, label_profiles):
    total_points = sum(len(group["points"]) for group in prompt_groups.values())
    total_boxes = sum(len(group["boxes"]) for group in prompt_groups.values())
    print(
        "[INFO] Prompt audit: "
        f"{len(label_order)} labels, {len(prompt_groups)} label/frame group(s), "
        f"{total_points} point(s), {total_boxes} box(es)."
    )

    if label_profiles:
        profile_names = sorted(set(label_profiles.keys()) & set(label_order.keys()))
        if profile_names:
            print(
                "[INFO] Label profiles loaded as operator guidance only: "
                + ", ".join(profile_names)
            )

    detail_count = 0
    max_details = 30
    sorted_groups = sorted(
        prompt_groups.items(),
        key=lambda item: (label_order.get(item[0][0], len(label_order)), item[0][1]),
    )

    for (label, frame_idx), group in sorted_groups:
        point_count = len(group["points"])
        box_count = len(group["boxes"])
        positives = sum(1 for value in group["labels"] if value == 1)
        negatives = point_count - positives

        if box_count > 1:
            raise ValueError(
                f"Label '{label}' frame {frame_idx} has {box_count} boxes. "
                "SAM2 accepts one box per object/frame prompt group."
            )

        if point_count > 1 or box_count:
            if detail_count < max_details:
                print(
                    "[WARN] Prompt audit: "
                    f"{label} frame {frame_idx} will be submitted as one grouped prompt "
                    f"({point_count} point(s): {positives} positive, {negatives} negative; "
                    f"{box_count} box(es))."
                )
                detail_count += 1
            elif detail_count == max_details:
                print("[WARN] Prompt audit: additional grouped prompt details suppressed.")
                detail_count += 1

        if point_count >= 3 and negatives == 0 and detail_count < max_details:
            print(
                "[WARN] Prompt audit: "
                f"{label} frame {frame_idx} has {point_count} positive point(s) and no negatives. "
                "Add negative clicks on excluded boundaries when precision matters."
            )
            detail_count += 1

def add_grouped_prompts(predictor, inference_state, prompt_groups, name_to_obj_id):
    for (label, frame_idx), group in prompt_groups.items():
        obj_id = name_to_obj_id[label]
        points = np.array(group["points"], dtype=np.float32) if group["points"] else None
        labels = np.array(group["labels"], dtype=np.int32) if group["labels"] else None
        box = group["boxes"][0] if group["boxes"] else None

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box,
        )

def io_consumer_thread(task_queue, output_path, masks_folder, width, height, fps, obj_id_to_name):
    """
    Background worker that handles saving mask files and encoding the video.
    This offloads heavy I/O and encoding from the main SAM2 propagation loop.
    """
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        print(f"[ERROR] Invalid dimensions for video writer: {w}x{h}")
        return

    # Use cv2.VideoWriter for better compatibility, matching segment_video.py
    # 'mp4v' is generally well-supported in headless environments
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not video_writer.isOpened():
        print(f"[ERROR] Failed to open VideoWriter for {output_path}")
        return

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    print(f"[INFO] I/O worker started. Output: {output_path} ({w}x{h} @ {fps}fps)")

    while True:
        task = task_queue.get()
        if task is None: break

        idx, out_obj_ids, masks_cpu, raw_frame_bytes = task
        
        # Height, Width, Channels for numpy/OpenCV
        try:
            processed_frame = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape((h, w, 3)).copy()
        except Exception as e:
            print(f"[ERROR] Failed to reshape frame {idx}: {e}")
            continue

        frame_masks = {}
        for i, obj_id in enumerate(out_obj_ids):
            mask_data = masks_cpu[i]

            # Ensure mask data is 2D (H, W)
            if mask_data.ndim == 3 and mask_data.shape[0] == 1:
                mask_data = mask_data.squeeze(0)
            elif mask_data.ndim == 3:
                mask_data = mask_data[0]

            # Handle resizing if necessary (e.g. if SAM2 output is not at original resolution)
            if mask_data.shape != (h, w):
                if mask_data.size > 0:
                    try:
                        # OpenCV resize takes (width, height)
                        mask_data = cv2.resize(mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    except Exception as e:
                        print(f"[ERROR] Resize failed for frame {idx}, obj {obj_id}: {e}")
                        mask_data = np.zeros((h, w), dtype=bool)
                else:
                    mask_data = np.zeros((h, w), dtype=bool)

            # Store for .npz saving
            label = obj_id_to_name.get(obj_id, f"obj_{obj_id}")
            frame_masks[label] = mask_data
            
            # Apply color overlay directly on numpy array (much faster than Matplotlib)
            color = colors[int(obj_id) % len(colors)]
            processed_frame[mask_data] = (processed_frame[mask_data] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        # Save masks for this frame
        np.savez_compressed(os.path.join(masks_folder, f"frame_{idx:05d}.npz"), **frame_masks)
        
        # Write to video (OpenCV expects BGR)
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(processed_frame_bgr)

        # Optional progress log every 100 frames
        if idx % 100 == 0:
            print(f"[INFO] Processed frame {idx}...")

    video_writer.release()
    print(f"[INFO] I/O worker finished.")

def load_label_mapping(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        coords = json.load(f)

    skipped_keys = [key for key, value in coords.items() if not isinstance(value, list)]
    if skipped_keys:
        print("[WARN] Ignoring non-label prompt JSON keys: " + ", ".join(skipped_keys))

    coords = {label: prompts for label, prompts in coords.items() if isinstance(prompts, list)}
    labels = list(coords.keys())
    for pts in coords.values():
        for p in pts:
            p["frame"] = normalize_frame_index(p.get("frame", 0))
    return {i: label for i, label in enumerate(labels)}, {label: i for i, label in enumerate(labels)}, coords

def main():
    config_path = "/config/config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        return

    with open(config_path) as f: 
        cfg = yaml.safe_load(f)
    
    video_dir = os.path.join(cfg["docker_data_dir"], cfg["frames_dir"])
    output_dir = os.path.join(cfg["docker_data_dir"], cfg["output_dir"])
    masks_folder = os.path.join(output_dir, "masks")
    os.makedirs(masks_folder, exist_ok=True)
    
    obj_id_to_name, name_to_obj_id, coords = load_label_mapping(os.path.join(cfg["docker_data_dir"], cfg["prompt_coordinates"]))
    label_profiles = cfg.get("label_profiles", {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Optimization: High precision matmul for newer GPUs
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    
    # Build predictor
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", 
        "/app/sam2/checkpoints/sam2.1_hiera_large.pt", 
        device=device
    )
    
    # Optimization: Compile the image encoder (can take a few minutes the first time)
    if hasattr(torch, "compile"):
        print("[INFO] Compiling image encoder for speed...")
        predictor.image_encoder = torch.compile(predictor.image_encoder, mode="reduce-overhead")
    
    # Initialize state
    inference_state = predictor.init_state(
        video_path=video_dir, 
        async_loading_frames=True, 
        offload_video_to_cpu=True
    )
    
    # Add prompts
    print("[INFO] Adding prompts to SAM2...")
    prompt_groups = build_prompt_groups(coords)
    print_prompt_audit(prompt_groups, name_to_obj_id, label_profiles)
    add_grouped_prompts(predictor, inference_state, prompt_groups, name_to_obj_id)
            
    # Get original frame dimensions
    frame_names = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.jpeg'))], key=lambda x: int(os.path.splitext(x)[0]))
    if not frame_names:
        print(f"[ERROR] No frames found in {video_dir}")
        return

    first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    w, h = first_frame.size
    print(f"[INFO] Original video resolution: {w}x{h}")
    
    # Start background I/O thread
    task_queue = queue.Queue(maxsize=30)
    output_video_path = os.path.join(output_dir, "output_video.mp4")
    io_thread = threading.Thread(
        target=io_consumer_thread, 
        args=(task_queue, output_video_path, masks_folder, w, h, cfg["fps"], obj_id_to_name)
    )
    io_thread.start()
    
    # Propagation loop
    print("[INFO] Starting video propagation...")
    with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
        for idx, obj_ids, logits in predictor.propagate_in_video(inference_state):
            # Convert logits to boolean masks on CPU
            masks_cpu = (logits > 0.0).cpu().numpy()
            
            # Load frame and pass to I/O thread as bytes to keep memory usage stable
            frame_path = os.path.join(video_dir, frame_names[idx])
            frame = (np.array(Image.open(frame_path))).astype(np.uint8)
            
            # Queue task
            task_queue.put((idx, obj_ids, masks_cpu, frame.tobytes()))
            
    # Signal I/O thread to finish
    task_queue.put(None)
    io_thread.join()
    print("[INFO] Video propagation complete.")

if __name__ == "__main__":
    main()
