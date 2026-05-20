import os
import torch
import time
import json
import argparse
import cv2
import numpy as np
import sys
import threading
import queue
import subprocess
from PIL import Image

# Ensure absolute pathing for project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from sam2.build_sam import build_sam2_video_predictor

def io_consumer_thread(task_queue, output_path, masks_folder, width, height, fps):
    # Ensure width and height are valid Python integers
    w = int(width)
    h = int(height)
    
    if w <= 0 or h <= 0:
        print(f"[I/O Worker] ERROR: Invalid dimensions - width={w}, height={h}")
        return
    
    print(f"[I/O Worker] Starting NVENC FFmpeg subprocess with dimensions {w}x{h}...")
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p', output_path
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Overlay colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        idx, out_obj_ids, masks_cpu, raw_frame = task
        
        frame_masks = {}
        processed_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3)).copy()
        
        for i, obj_id in enumerate(out_obj_ids):
            # Mask is already a boolean numpy array from the main thread
            mask_data = masks_cpu[i]

            # If mask shape doesn't match frame, resize it
            if mask_data.shape != (h, w):
                try:
                    # SAM2 logits are typically (1, H, W) or (H, W). Remove singleton dim if present.
                    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
                        mask_data = mask_data.squeeze(0)
                    
                    # Log shape for debugging if it still doesn't match
                    if mask_data.shape != (h, w):
                        mask_data = cv2.resize(mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                except Exception as e:
                    print(f"[I/O Worker] Resize error: {e}. Mask shape: {mask_data.shape}, Target: {(w, h)}")
                    raise e
            
            frame_masks[str(obj_id)] = mask_data
            
            # Apply overlay
            color = colors[int(obj_id) % len(colors)]
            processed_frame[mask_data] = (processed_frame[mask_data] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            
        np.savez(os.path.join(masks_folder, f"frame_{idx:05d}.npz"), **frame_masks)
        proc.stdin.write(processed_frame.tobytes())
    
    print("[I/O Worker] Closing encoder...")
    proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait()
    print("[I/O Worker] Process finished.")

def run_benchmark(config, checkpoint, project, id, project_json_path, max_frames, local_frames=None):
    OUTPUT_BASE = r"Z:\Videos\MCD\sam2_outputs\exp"
    output_dir = os.path.join(OUTPUT_BASE, project, id, f"benchmark_{max_frames}frames")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(project_json_path, 'r') as f:
        projects_data = json.load(f)['projects']
    project_info = next((p for p in projects_data if p['name'] == project), None)
    data = project_info['data'][id]
    
    frame_dir = os.path.abspath(local_frames) if local_frames else data['frames']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    
    predictor = build_sam2_video_predictor(os.path.abspath(config), os.path.abspath(checkpoint), device=device)
    
    # Apply safe partial compilation to the static backbone
    predictor.image_encoder = torch.compile(predictor.image_encoder, mode="reduce-overhead")
    
    # Wrap lifecycle in autocast
    with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
        inference_state = predictor.init_state(
            video_path=frame_dir, async_loading_frames=True,
            offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        predictor.reset_state(inference_state)
        
        with open(data['prompts'], 'r') as f:
            prompts = json.load(f)
            
        # Batch points by frame
        frame_points = {}
        for label, points_list in prompts.items():
            obj_id = list(prompts.keys()).index(label)
            for p in points_list:
                if p['frame'] < max_frames:
                    f_idx = p['frame']
                    if f_idx not in frame_points: frame_points[f_idx] = {}
                    if obj_id not in frame_points[f_idx]: frame_points[f_idx][obj_id] = {'coords': [], 'labels': []}
                    frame_points[f_idx][obj_id]['coords'].append([p['x'], p['y']])
                    frame_points[f_idx][obj_id]['labels'].append(p['positive'])

        for f_idx in sorted(frame_points.keys()):
            for obj_id, data_pts in frame_points[f_idx].items():
                predictor.add_new_points_or_box(
                    inference_state=inference_state, 
                    frame_idx=f_idx, 
                    obj_id=obj_id, 
                    points=np.array(data_pts['coords'], dtype=np.float32), 
                    labels=np.array(data_pts['labels'], dtype=np.int32)
                )

        masks_folder = os.path.join(output_dir, "masks")
        os.makedirs(masks_folder, exist_ok=True)
        
        # Get and validate frame dimensions (convert to Python ints)
        h, w = inference_state['images'][0].shape[1:3]
        height = int(h)
        width = int(w)
        
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid frame dimensions from inference state: height={height}, width={width}")
        
        print(f"Frame dimensions: {width}x{height}")
        
        task_queue = queue.Queue(maxsize=30)
        io_thread = threading.Thread(target=io_consumer_thread, args=(task_queue, os.path.join(output_dir, "output.mp4"), masks_folder, width, height, 30))
        io_thread.daemon = True
        io_thread.start()

        times = {}
        prop_start = time.perf_counter()
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx >= max_frames: break
            
            # 1. Evaluate mask and move to CPU immediately to free VRAM
            # out_mask_logits is shaped [N, 1, H, W] or [N, H, W]
            masks_cpu = (out_mask_logits > 0.0).cpu().numpy()

            # 2. Fix scaling bug: multiply normalized float by 255
            frame_tensor = inference_state['images'][out_frame_idx]
            raw_frame = (frame_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

            # Push safe CPU data to the queue
            task_queue.put((out_frame_idx, out_obj_ids, masks_cpu, raw_frame.tobytes()))

        task_queue.put(None)
        io_thread.join()
        
        times['inference_propagation'] = time.perf_counter() - prop_start
    
    with open(os.path.join(output_dir, "timing_report.json"), "w") as f:
        json.dump(times, f, indent=4)
    print(f"[Benchmark] Finished. Time: {times['inference_propagation']:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--local-frames", type=str, default=None)
    args = parser.parse_args()
    # Fix: Point to the correct projects.json location
    PROJECTS_JSON = os.path.join(BASE_DIR, "projects.json")
    run_benchmark(args.config, args.checkpoint, args.project, args.id, PROJECTS_JSON, args.frames, args.local_frames)
