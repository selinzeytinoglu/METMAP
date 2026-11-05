import os
import yaml
import subprocess


def main():
    config_path = "/config/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    video_path  = os.path.join(cfg["docker_data_dir"], cfg["video_file"])
    frames_dir  = os.path.join(cfg["docker_data_dir"], cfg["frames_dir"])
    start_frame = cfg["start_frame"]
    end_frame   = cfg["end_frame"]

    os.makedirs(frames_dir, exist_ok=True)

    select_filter = f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS"

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", select_filter,
        "-q:v", "2",
        "-start_number", "0",
        os.path.join(frames_dir, "%05d.jpg")
    ], check=True)

    print(f"Extracted frames {start_frame}–{end_frame} to {frames_dir}")


if __name__ == "__main__":
    main()
