import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml

# Import SAM2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import build_sam2_video_predictor


def cfg_get(cfg, key, default):
    return cfg[key] if key in cfg else default


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
                raise ValueError(
                    f"Point prompt for label '{label}' frame {frame_idx} is missing x/y: {prompt}"
                )

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


def load_label_mapping(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        coords = json.load(f)

    skipped_keys = [key for key, value in coords.items() if not isinstance(value, list)]
    if skipped_keys:
        print("[WARN] Ignoring non-label prompt JSON keys: " + ", ".join(skipped_keys))

    coords = {
        label: prompts for label, prompts in coords.items() if isinstance(prompts, list)
    }
    labels = list(coords.keys())
    for pts in coords.values():
        for p in pts:
            p["frame"] = normalize_frame_index(p.get("frame", 0))
    return {i: label for i, label in enumerate(labels)}, {
        label: i for i, label in enumerate(labels)
    }, coords


def list_frame_paths(video_dir):
    frame_names = [
        f
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[-1].lower() in (".jpg", ".jpeg")
    ]
    frame_names.sort(key=normalize_frame_index)
    if not frame_names:
        raise RuntimeError(f"No JPG/JPEG frames found in {video_dir}")
    return [os.path.join(video_dir, name) for name in frame_names]


def read_bgr_frame(path, expected_width=None, expected_height=None):
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {path}")

    if expected_width is not None and expected_height is not None:
        height, width = frame.shape[:2]
        if width != expected_width or height != expected_height:
            frame = cv2.resize(
                frame, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR
            )
    return frame


class OriginalFramePrefetcher:
    """Sequential OpenCV frame reader for overlay frames."""

    def __init__(self, frame_paths, width, height, queue_size):
        self.frame_paths = frame_paths
        self.width = int(width)
        self.height = int(height)
        self.queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            for idx, path in enumerate(self.frame_paths):
                frame = read_bgr_frame(path, self.width, self.height)
                self.queue.put((idx, frame))
        except BaseException as exc:
            self.queue.put(exc)
        finally:
            self.queue.put(None)

    def get(self, expected_idx):
        item = self.queue.get()
        if item is None:
            raise RuntimeError(
                f"Frame reader ended before expected frame {expected_idx} was available."
            )
        if isinstance(item, BaseException):
            raise RuntimeError("Frame reader failed") from item

        idx, frame = item
        if idx != expected_idx:
            raise RuntimeError(f"Expected frame {expected_idx}, but reader returned {idx}")
        return frame

    def join(self):
        self.thread.join(timeout=5)


class Sam2FramePrefetcher:
    """
    Restores bounded async loading for SAM2's lazy JPEG loader.

    The local AsyncVideoFrameLoader has its own background loader and cache disabled,
    so propagate_in_video otherwise pays PIL decode/resize cost on the main GPU path.
    This wrapper fills the existing loader cache a small window ahead of inference.
    """

    def __init__(self, image_loader, lookahead=24, retain=2, enabled=True):
        self.image_loader = image_loader
        self.lookahead = max(0, int(lookahead))
        self.retain = max(0, int(retain))
        self.enabled = (
            bool(enabled)
            and self.lookahead > 0
            and hasattr(image_loader, "images")
            and hasattr(image_loader, "img_paths")
        )
        self.condition = threading.Condition()
        self.consumed_idx = -1
        self.next_idx = 0
        self.cleared_until = 0
        self.stop_requested = False
        self.exception = None
        self.thread = None

    def start(self):
        if not self.enabled:
            print("[INFO] SAM2 frame prefetch disabled or not applicable.")
            return
        print(
            "[INFO] SAM2 frame prefetch enabled: "
            f"lookahead={self.lookahead}, retain={self.retain}."
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            frame_count = len(self.image_loader.images)
            while True:
                with self.condition:
                    while (
                        not self.stop_requested
                        and self.next_idx > self.consumed_idx + self.lookahead
                    ):
                        self.condition.wait(timeout=0.05)

                    if self.stop_requested or self.next_idx >= frame_count:
                        return

                    idx = self.next_idx
                    self.next_idx += 1

                if self.image_loader.images[idx] is None:
                    image = self.image_loader.__getitem__(idx)
                    if isinstance(image, torch.Tensor):
                        image = image.to(dtype=torch.float16)
                    self.image_loader.images[idx] = image
        except BaseException as exc:
            with self.condition:
                self.exception = exc
                self.stop_requested = True
                self.condition.notify_all()

    def mark_consumed(self, idx):
        if not self.enabled:
            return

        clear_start = None
        clear_end = None
        with self.condition:
            if self.exception is not None:
                raise RuntimeError("SAM2 frame prefetch failed") from self.exception

            self.consumed_idx = max(self.consumed_idx, int(idx))
            clear_end = max(0, self.consumed_idx - self.retain)
            if clear_end > self.cleared_until:
                clear_start = self.cleared_until
                self.cleared_until = clear_end
            self.condition.notify_all()

        if clear_start is not None:
            for old_idx in range(clear_start, clear_end):
                self.image_loader.images[old_idx] = None

    def stop(self):
        if not self.enabled:
            return
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=5)


class NullVideoSink:
    def write(self, frame_bgr):
        return

    def close(self):
        return


class OpenCVVideoSink:
    def __init__(self, output_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_path, fourcc, float(fps), (int(width), int(height))
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open OpenCV VideoWriter for {output_path}")
        print(f"[INFO] Using OpenCV mp4v video encoder: {output_path}")

    def write(self, frame_bgr):
        self.writer.write(frame_bgr)

    def close(self):
        self.writer.release()


class FFmpegNvencVideoSink:
    def __init__(self, output_path, width, height, fps):
        self.output_path = output_path
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        print(f"[INFO] Using FFmpeg h264_nvenc video encoder: {output_path}")

    def write(self, frame_bgr):
        if self.proc.poll() is not None:
            stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg exited early while writing video: {stderr}")
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except (BrokenPipeError, OSError) as exc:
            stderr = ""
            if self.proc.poll() is not None:
                stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"FFmpeg pipe failed while writing video: {stderr}"
            ) from exc

    def close(self):
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
        return_code = self.proc.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr}")


def ffmpeg_supports_encoder(encoder_name):
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False
    return encoder_name in result.stdout


def create_video_sink(encoder, output_path, width, height, fps, write_video):
    if not write_video:
        print("[INFO] Overlay video writing disabled.")
        return NullVideoSink()

    encoder = str(encoder).lower()
    if encoder in ("auto", "ffmpeg_nvenc"):
        if ffmpeg_supports_encoder("h264_nvenc"):
            try:
                return FFmpegNvencVideoSink(output_path, width, height, fps)
            except Exception as exc:
                if encoder == "ffmpeg_nvenc":
                    raise
                print(f"[WARN] Could not start NVENC encoder; falling back to OpenCV: {exc}")
        elif encoder == "ffmpeg_nvenc":
            raise RuntimeError("ffmpeg is not available with h264_nvenc support.")

    return OpenCVVideoSink(output_path, width, height, fps)


def normalize_mask(mask_data, width, height):
    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
        mask_data = mask_data.squeeze(0)
    elif mask_data.ndim == 3:
        mask_data = mask_data[0]

    if mask_data.shape != (height, width):
        if mask_data.size == 0:
            return np.zeros((height, width), dtype=bool)
        mask_data = cv2.resize(
            mask_data.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask_data.astype(bool, copy=False)


def normalize_mask_storage(mask_storage, compress_masks):
    if mask_storage is None:
        return "compressed_bool" if compress_masks else "packbits"

    value = str(mask_storage).strip().lower().replace("-", "_")
    aliases = {
        "raw": "bool",
        "boolean": "bool",
        "uncompressed": "bool",
        "npz": "bool",
        "zlib": "compressed_bool",
        "compressed": "compressed_bool",
        "npz_compressed": "compressed_bool",
        "packed": "packbits",
        "bitpack": "packbits",
        "bit_packed": "packbits",
    }
    value = aliases.get(value, value)
    if value not in {"bool", "compressed_bool", "packbits"}:
        raise ValueError(
            "mask_storage must be one of: bool, compressed_bool, packbits"
        )
    return value


def save_frame_masks(mask_path, frame_masks, mask_storage, width, height):
    if mask_storage == "compressed_bool":
        np.savez_compressed(mask_path, **frame_masks)
        return

    if mask_storage == "bool":
        np.savez(mask_path, **frame_masks)
        return

    packed = {
        "__mask_storage__": np.array("packbits"),
        "__height__": np.array(int(height), dtype=np.int32),
        "__width__": np.array(int(width), dtype=np.int32),
        "__bitorder__": np.array("little"),
    }
    for label, mask in frame_masks.items():
        packed[label] = np.packbits(
            np.ascontiguousarray(mask, dtype=bool).reshape(-1),
            bitorder="little",
        )
    np.savez(mask_path, **packed)


def io_consumer_thread(
    task_queue,
    output_path,
    masks_folder,
    width,
    height,
    fps,
    obj_id_to_name,
    save_masks,
    mask_storage,
    video_encoder,
    write_overlay_video,
    error_holder,
):
    width = int(width)
    height = int(height)
    video_sink = None

    try:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid output dimensions: {width}x{height}")

        if save_masks:
            os.makedirs(masks_folder, exist_ok=True)

        video_sink = create_video_sink(
            video_encoder, output_path, width, height, fps, write_overlay_video
        )

        # BGR colors because frames are read by OpenCV and written without conversion.
        colors = [
            np.array((0, 0, 255), dtype=np.float32),
            np.array((0, 255, 0), dtype=np.float32),
            np.array((255, 0, 0), dtype=np.float32),
            np.array((0, 255, 255), dtype=np.float32),
            np.array((255, 0, 255), dtype=np.float32),
            np.array((255, 255, 0), dtype=np.float32),
        ]

        processed_count = 0
        while True:
            task = task_queue.get()
            if task is None:
                break

            idx, out_obj_ids, masks_cpu, frame_bgr = task
            processed_frame = frame_bgr.copy()
            frame_masks = {}

            for i, obj_id in enumerate(out_obj_ids):
                obj_id_int = int(obj_id)
                mask_data = normalize_mask(masks_cpu[i], width, height)

                if save_masks:
                    label = obj_id_to_name.get(obj_id_int, f"obj_{obj_id_int}")
                    frame_masks[label] = mask_data

                color = colors[obj_id_int % len(colors)]
                processed_frame[mask_data] = (
                    processed_frame[mask_data].astype(np.float32) * 0.5 + color * 0.5
                ).astype(np.uint8)

            if save_masks:
                mask_path = os.path.join(masks_folder, f"frame_{idx:05d}.npz")
                save_frame_masks(mask_path, frame_masks, mask_storage, width, height)

            video_sink.write(processed_frame)
            processed_count += 1
            if idx % 100 == 0:
                print(f"[INFO] I/O worker processed frame {idx}...")

        print(f"[INFO] I/O worker finished {processed_count} frame(s).")
    except BaseException as exc:
        error_holder.append(exc)
    finally:
        if video_sink is not None:
            try:
                video_sink.close()
            except BaseException as exc:
                error_holder.append(exc)


def raise_if_io_worker_failed(io_errors, io_thread):
    if io_errors:
        raise RuntimeError("I/O worker failed") from io_errors[0]
    if not io_thread.is_alive():
        raise RuntimeError(
            "I/O worker stopped before accepting all propagated frames."
        )


def enqueue_io_task(
    task_queue,
    task,
    io_errors,
    io_thread,
    max_wait_seconds,
):
    frame_idx = task[0] if task else "unknown"
    max_wait_seconds = max(0.1, float(max_wait_seconds))
    deadline = perf_counter() + max_wait_seconds

    while True:
        raise_if_io_worker_failed(io_errors, io_thread)
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError(
                "I/O worker did not accept "
                f"frame {frame_idx} within {max_wait_seconds:.1f}s. "
                "The output queue may be blocked by video encoding or disk writes."
            )

        try:
            task_queue.put(task, timeout=min(0.5, remaining))
            return
        except queue.Full:
            continue


def stop_io_worker(task_queue, io_thread, io_errors, max_wait_seconds):
    max_wait_seconds = max(0.1, float(max_wait_seconds))
    deadline = perf_counter() + max_wait_seconds
    sentinel_sent = False

    while io_thread.is_alive() and not sentinel_sent:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return TimeoutError(
                f"I/O worker did not accept shutdown within {max_wait_seconds:.1f}s."
            )

        try:
            task_queue.put(None, timeout=min(0.5, remaining))
            sentinel_sent = True
        except queue.Full:
            if io_errors and not io_thread.is_alive():
                break
            continue

    if io_thread.is_alive():
        io_thread.join(timeout=max(0.1, deadline - perf_counter()))

    if io_thread.is_alive():
        return TimeoutError(
            f"I/O worker did not stop within {max_wait_seconds:.1f}s."
        )

    return None


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/config/config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        return

    total_start = perf_counter()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["docker_data_dir"]
    video_dir = os.path.join(data_dir, cfg["frames_dir"])
    output_dir = os.path.join(data_dir, cfg["output_dir"])
    masks_folder = os.path.join(output_dir, "masks")
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = list_frame_paths(video_dir)
    first_frame = read_bgr_frame(frame_paths[0])
    height, width = first_frame.shape[:2]
    print(f"[INFO] Original video resolution: {width}x{height}")
    print(f"[INFO] Frame count: {len(frame_paths)}")

    obj_id_to_name, name_to_obj_id, coords = load_label_mapping(
        os.path.join(data_dir, cfg["prompt_coordinates"])
    )
    label_profiles = cfg_get(cfg, "label_profiles", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model_cfg = cfg_get(cfg, "sam2_model_config", "configs/sam2.1/sam2.1_hiera_l.yaml")
    sam2_checkpoint = cfg_get(
        cfg, "sam2_checkpoint", "/app/sam2/checkpoints/sam2.1_hiera_large.pt"
    )
    compile_image_encoder = bool(cfg_get(cfg, "compile_image_encoder", True))
    async_loading_frames = bool(cfg_get(cfg, "async_loading_frames", True))
    offload_video_to_cpu = bool(cfg_get(cfg, "offload_video_to_cpu", True))
    offload_state_to_cpu = bool(cfg_get(cfg, "offload_state_to_cpu", False))

    save_masks = bool(cfg_get(cfg, "save_masks", True))
    compress_masks = bool(cfg_get(cfg, "compress_masks", False))
    mask_storage = normalize_mask_storage(cfg_get(cfg, "mask_storage", None), compress_masks)
    write_overlay_video = bool(cfg_get(cfg, "write_overlay_video", True))
    video_encoder = cfg_get(cfg, "video_encoder", "auto")
    frame_queue_size = int(cfg_get(cfg, "frame_queue_size", 24))
    result_queue_size = int(cfg_get(cfg, "result_queue_size", 24))
    io_queue_put_timeout_seconds = float(
        cfg_get(cfg, "io_queue_put_timeout_seconds", 60.0)
    )
    io_thread_stop_timeout_seconds = float(
        cfg_get(cfg, "io_thread_stop_timeout_seconds", 300.0)
    )
    sam2_prefetch_enabled = bool(cfg_get(cfg, "sam2_frame_prefetch", True))
    sam2_prefetch_lookahead = int(cfg_get(cfg, "sam2_frame_prefetch_lookahead", 24))
    sam2_prefetch_retain = int(cfg_get(cfg, "sam2_frame_prefetch_retain", 2))

    print(
        "[INFO] Speed settings: "
        f"save_masks={save_masks}, mask_storage={mask_storage}, "
        f"video_encoder={video_encoder}, write_overlay_video={write_overlay_video}, "
        f"offload_video_to_cpu={offload_video_to_cpu}, "
        f"offload_state_to_cpu={offload_state_to_cpu}."
    )

    frame_prefetcher = OriginalFramePrefetcher(
        frame_paths, width, height, queue_size=frame_queue_size
    )
    frame_prefetcher.start()

    setup_start = perf_counter()
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    if compile_image_encoder and hasattr(torch, "compile"):
        print("[INFO] Compiling image encoder for speed...")
        predictor.image_encoder = torch.compile(
            predictor.image_encoder, mode="reduce-overhead"
        )

    with torch.inference_mode(), torch.autocast(
        "cuda", enabled=torch.cuda.is_available(), dtype=torch.float16
    ):
        inference_state = predictor.init_state(
            video_path=video_dir,
            async_loading_frames=async_loading_frames,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )

        sam2_prefetcher = Sam2FramePrefetcher(
            inference_state["images"],
            lookahead=sam2_prefetch_lookahead,
            retain=sam2_prefetch_retain,
            enabled=sam2_prefetch_enabled,
        )
        sam2_prefetcher.start()

        print("[INFO] Adding prompts to SAM2...")
        prompt_groups = build_prompt_groups(coords)
        print_prompt_audit(prompt_groups, name_to_obj_id, label_profiles)
        add_grouped_prompts(predictor, inference_state, prompt_groups, name_to_obj_id)

        setup_seconds = perf_counter() - setup_start
        print(f"[INFO] Setup complete in {setup_seconds:.2f}s.")

        task_queue = queue.Queue(maxsize=max(1, result_queue_size))
        output_video_path = os.path.join(output_dir, "output_video.mp4")
        io_errors = []
        io_thread = threading.Thread(
            target=io_consumer_thread,
            args=(
                task_queue,
                output_video_path,
                masks_folder,
                width,
                height,
                cfg["fps"],
                obj_id_to_name,
                save_masks,
                mask_storage,
                video_encoder,
                write_overlay_video,
                io_errors,
            ),
            daemon=True,
        )
        io_thread.start()

        print("[INFO] Starting video propagation...")
        propagation_start = perf_counter()
        frame_count = 0
        io_stop_error = None
        try:
            for idx, obj_ids, logits in predictor.propagate_in_video(inference_state):
                masks_cpu = (logits > 0.0).cpu().numpy()
                frame_bgr = frame_prefetcher.get(idx)
                enqueue_io_task(
                    task_queue,
                    (idx, obj_ids, masks_cpu, frame_bgr),
                    io_errors,
                    io_thread,
                    io_queue_put_timeout_seconds,
                )
                sam2_prefetcher.mark_consumed(idx)
                frame_count += 1
        finally:
            io_stop_error = stop_io_worker(
                task_queue,
                io_thread,
                io_errors,
                io_thread_stop_timeout_seconds,
            )
            sam2_prefetcher.stop()
            frame_prefetcher.join()

        if io_errors:
            raise RuntimeError("I/O worker failed") from io_errors[0]
        if io_stop_error is not None:
            raise io_stop_error

    propagation_seconds = perf_counter() - propagation_start
    total_seconds = perf_counter() - total_start
    fps = frame_count / propagation_seconds if propagation_seconds > 0 else 0.0
    end_to_end_fps = frame_count / total_seconds if total_seconds > 0 else 0.0

    timing = {
        "frames": frame_count,
        "setup_seconds": setup_seconds,
        "propagation_and_output_seconds": propagation_seconds,
        "total_seconds": total_seconds,
        "propagation_and_output_fps": fps,
        "end_to_end_fps": end_to_end_fps,
        "settings": {
            "compile_image_encoder": compile_image_encoder,
            "async_loading_frames": async_loading_frames,
            "sam2_frame_prefetch": sam2_prefetch_enabled,
            "sam2_frame_prefetch_lookahead": sam2_prefetch_lookahead,
            "sam2_frame_prefetch_retain": sam2_prefetch_retain,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "save_masks": save_masks,
            "mask_storage": mask_storage,
            "write_overlay_video": write_overlay_video,
            "video_encoder": video_encoder,
            "frame_queue_size": frame_queue_size,
            "result_queue_size": result_queue_size,
            "io_queue_put_timeout_seconds": io_queue_put_timeout_seconds,
            "io_thread_stop_timeout_seconds": io_thread_stop_timeout_seconds,
        },
    }
    timing_path = os.path.join(output_dir, "timing_report_even_faster.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    print(
        "[INFO] Video propagation complete. "
        f"{frame_count} frame(s), {propagation_seconds:.2f}s, {fps:.2f} FPS "
        f"({total_seconds:.2f}s end-to-end)."
    )
    print(f"[INFO] Timing report written to {timing_path}")


if __name__ == "__main__":
    main()
