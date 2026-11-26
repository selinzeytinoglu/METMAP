import os
import yaml
import subprocess


def main():
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    config_path = "/config/config.yaml" if os.path.isfile("/config/config.yaml") else default_cfg
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["docker_data_dir"] if os.path.isdir(cfg["docker_data_dir"]) else cfg["local_data_dir"]
    video_path  = os.path.join(data_dir, cfg["video_file"])
    frames_dir  = os.path.join(data_dir, cfg["frames_dir"])
    start_frame = int(cfg.get("start_frame") or 0)
    end_frame   = cfg.get("end_frame")

    os.makedirs(frames_dir, exist_ok=True)

    if end_frame is None or str(end_frame).strip() == "":
        select_filter = f"select='gte(n,{start_frame})',setpts=PTS-STARTPTS"
        range_label = f"{start_frame}-end"
    else:
        end_frame = int(end_frame)
        select_filter = f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS"
        range_label = f"{start_frame}-{end_frame}"

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", select_filter,
        "-vsync", "0",
        "-q:v", "2",
        "-start_number", "0",
        os.path.join(frames_dir, "%05d.jpg")
    ], check=True)

    print(f"Extracted frames {range_label} to {frames_dir}")


if __name__ == "__main__":
    main()
