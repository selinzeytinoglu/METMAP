import os
import sys
import yaml
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import perf_counter
from PIL import Image
import json
import cv2
import io

# ffmpeg -i "Z:\IndividualStudies\1. Mother Child Dynamics (F32) study\Data\3. ET Data\Child Raw Files\571_Child\Speech Task\exports\000\571_Child_Speech_Corrected.mp4" -vf "select=between(n\,3897\,5396),setpts=PTS-STARTPTS" -q:v 2 -start_number 0 C:\Users\abudlong\Desktop\frames_571/'%05d.jpg'


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def fig2img(fig):
    # Convert a Matplotlib figure to a numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def load_label_mapping(prompt_coordinates_path):
    with open(prompt_coordinates_path, 'r') as f:
        coordinates = json.load(f)
    labels = list(coordinates.keys())
    obj_id_to_name = {i: label for i, label in enumerate(labels)}
    name_to_obj_id = {label: i for i, label in enumerate(labels)}
    return obj_id_to_name, name_to_obj_id, coordinates


def add_clicks(predictor, inference_state, ann_frame_idx, ann_obj_id, points, labels):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    return out_obj_ids, out_mask_logits


def prompt_sam2(coordinates, name_to_obj_id, inference_state, predictor):
    for label, coords in coordinates.items():
        ann_obj_id = name_to_obj_id[label]
        for coord in coords:
            print(f"({coord['x']},{coord['y']}) Frame: {coord['frame']} Positive?: {coord['positive']}")
            points = np.array([[coord['x'], coord['y']]], dtype=np.float32)
            labels = np.array([coord['positive']], np.int32)
            add_clicks(predictor, inference_state, coord['frame'], ann_obj_id, points, labels)


def main():
    config_path = "/config/config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def p(relative_path):
        return os.path.join(cfg["docker_data_dir"], relative_path)

    video_dir = p(cfg["frames_dir"])
    prompt_coordinates = p(cfg["prompt_coordinates"])
    output_dir = p(cfg["output_dir"])
    fps = cfg["fps"]
    sam2_checkpoint = "/app/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    device = setup_device()

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sam2.build_sam import build_sam2_video_predictor

    obj_id_to_name, name_to_obj_id, coordinates = load_label_mapping(prompt_coordinates)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    width, height = first_frame.size

    masks_folder_path = os.path.join(output_dir, "masks")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_folder_path, exist_ok=True)

    output_video_path = os.path.join(output_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(
        video_path=video_dir,
        async_loading_frames=True,
        offload_state_to_cpu=False,
        offload_video_to_cpu=True
    )

    prompt_sam2(coordinates, name_to_obj_id, inference_state, predictor)

    tprev = -1

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if (perf_counter() > tprev + 1.0) and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            print("VRAM:", (total_bytes - free_bytes) // 1_000_000, "MB")
            tprev = perf_counter()

        # Save masks for this frame
        frame_masks = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            if out_mask.shape[0] == 1 and len(out_mask.shape) == 3:
                out_mask = np.squeeze(out_mask, axis=0)
            frame_masks[obj_id_to_name[out_obj_id]] = out_mask

        mask_path = os.path.join(masks_folder_path, f"frame_{out_frame_idx:05d}.npz")
        np.savez_compressed(mask_path, **frame_masks)
        print(mask_path)

        # Render frame with mask overlays and write to output video
        fig = plt.figure(figsize=(width/100, height/100), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        frame = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        ax.imshow(frame)

        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            show_mask(out_mask, ax, obj_id=out_obj_id)

        img = fig2img(fig)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
        plt.close(fig)

    video_writer.release()


if __name__ == "__main__":
    main()
