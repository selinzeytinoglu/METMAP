import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np

MASK_METADATA_PREFIX = "__"


def normalize_frame_index(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return int(stem)

    digits = "".join(ch for ch in reversed(stem) if ch.isdigit())
    if digits:
        return int(digits[::-1])

    raise ValueError(f"Cannot parse frame index from '{path}'")


def list_indexed_files(directory, extensions):
    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name)[-1].lower() in extensions
    ]
    files.sort(key=normalize_frame_index)
    return files


def npz_scalar_to_str(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_masks(mask_path):
    with np.load(mask_path) as masks_data:
        keys = [
            key for key in masks_data.files if not key.startswith(MASK_METADATA_PREFIX)
        ]
        storage = "bool"
        if "__mask_storage__" in masks_data.files:
            storage = npz_scalar_to_str(masks_data["__mask_storage__"])

        if storage == "packbits":
            height = int(masks_data["__height__"])
            width = int(masks_data["__width__"])
            bitorder = npz_scalar_to_str(masks_data.get("__bitorder__", "little"))
            count = height * width
            return {
                key: np.unpackbits(masks_data[key], bitorder=bitorder, count=count)
                .reshape(height, width)
                .astype(bool)
                for key in keys
            }

        return {key: masks_data[key].astype(bool) for key in keys}


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


def create_video_sink(encoder, output_path, width, height, fps):
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


def overlay_masks(frame_bgr, masks, label_order, alpha):
    colors = [
        np.array((0, 0, 255), dtype=np.float32),
        np.array((0, 255, 0), dtype=np.float32),
        np.array((255, 0, 0), dtype=np.float32),
        np.array((0, 255, 255), dtype=np.float32),
        np.array((255, 0, 255), dtype=np.float32),
        np.array((255, 255, 0), dtype=np.float32),
    ]
    height, width = frame_bgr.shape[:2]
    output = frame_bgr.copy()

    for label, mask in masks.items():
        if label not in label_order:
            label_order.append(label)
        color = colors[label_order.index(label) % len(colors)]
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        output[mask] = (
            output[mask].astype(np.float32) * (1.0 - alpha) + color * alpha
        ).astype(np.uint8)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Render an overlay video from saved SAM2 mask .npz files."
    )
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    frame_paths = list_indexed_files(args.frames_dir, {".jpg", ".jpeg"})
    mask_paths = list_indexed_files(args.masks_dir, {".npz"})
    if not frame_paths:
        raise RuntimeError(f"No JPG/JPEG frames found in {args.frames_dir}")
    if not mask_paths:
        raise RuntimeError(f"No .npz mask files found in {args.masks_dir}")
    if len(frame_paths) != len(mask_paths):
        raise RuntimeError(
            f"Frame/mask count mismatch: {len(frame_paths)} frame(s), "
            f"{len(mask_paths)} mask file(s)."
        )

    first_frame = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sink = create_video_sink(args.video_encoder, args.output, width, height, args.fps)

    label_order = []
    try:
        for idx, (frame_path, mask_path) in enumerate(zip(frame_paths, mask_paths)):
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            masks = load_masks(mask_path)
            sink.write(overlay_masks(frame, masks, label_order, args.alpha))
            if idx % 100 == 0:
                print(f"[INFO] Rendered frame {idx}...")
    finally:
        sink.close()

    print(f"[INFO] Rendered {len(frame_paths)} frame(s) to {args.output}")


if __name__ == "__main__":
    main()
