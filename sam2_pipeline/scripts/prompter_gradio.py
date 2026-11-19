import argparse
from contextlib import nullcontext
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import json
import os
import sys
import time
import yaml

# Fixed display resolution. The gr.Image component height is set to match
# DISPLAY_H so the browser renders the image at exactly this size.
DISPLAY_W = 1280
DISPLAY_H = 720
DOT_RADIUS = 4

SAM2_MODEL_CHOICES = [
    "facebook/sam2.1-hiera-large",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-tiny",
]
ROI_PALETTE = [
    (46, 204, 113),
    (52, 152, 219),
    (231, 76, 60),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
]
_IMAGE_PREDICTORS = {}


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_image_predictor(model_id):
    if model_id not in _IMAGE_PREDICTORS:
        pipeline_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        if pipeline_dir not in sys.path:
            sys.path.append(pipeline_dir)
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = get_device()
        cfg_And_ckpt = HF_MODEL_ID_TO_FILENAMES.get(model_id)
        local_ckpt = None
        if cfg_And_ckpt is not None:
            cfg_name, ckpt_filename = cfg_And_ckpt
            candidate_dirs = [
                "/app/sam2/checkpoints",
                os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sam2", "checkpoints")),
            ]
            for d in candidate_dirs:
                p = os.path.join(d, ckpt_filename)
                if os.path.isfile(p):
                    local_ckpt = (cfg_name, p)
                    break

        if local_ckpt is not None:
            sam_model = build_sam2(local_ckpt[0], local_ckpt[1], device=device)
            _IMAGE_PREDICTORS[model_id] = SAM2ImagePredictor(sam_model)
        else:
            _IMAGE_PREDICTORS[model_id] = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    return _IMAGE_PREDICTORS[model_id]


# Helpers

def get_image_files(folder_path):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
    files.sort(key=lambda f: (0, int(os.path.splitext(f)[0])) if os.path.splitext(f)[0].isdigit() else (1, f))
    return files


def resize_to_display(orig_img):
    # Resize a PIL image to fit within DISPLAY_W x DISPLAY_H maintaining aspect ratio
    orig_w, orig_h = orig_img.size
    scale = min(DISPLAY_W / orig_w, DISPLAY_H / orig_h)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    return orig_img.resize((disp_w, disp_h), Image.LANCZOS), orig_w, orig_h, disp_w, disp_h


def get_roi_color(label, label_order=None):
    if label_order and label in label_order:
        idx = label_order.index(label)
    else:
        idx = sum(ord(c) for c in label)
    return ROI_PALETTE[idx % len(ROI_PALETTE)]


def overlay_masks(display_img, frame_masks, label_order=None, alpha=0.45):
    if not frame_masks:
        return display_img
    disp_w, disp_h = display_img.size
    base = np.array(display_img, dtype=np.float32)
    ordered = [l for l in (label_order or []) if l in frame_masks] + [
        l for l in frame_masks.keys() if not label_order or l not in label_order
    ]
    for label in ordered:
        mask = frame_masks.get(label)
        if mask is None:
            continue
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((disp_w, disp_h), Image.NEAREST)
        m = np.array(mask_img) > 127
        if not np.any(m):
            continue
        color = np.array(get_roi_color(label, label_order), dtype=np.float32)
        base[m] = base[m] * (1.0 - alpha) + color * alpha
        edge = m & ~(
            np.roll(m, 1, axis=0) & np.roll(m, -1, axis=0) &
            np.roll(m, 1, axis=1) & np.roll(m, -1, axis=1)
        )
        base[edge] = color
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _parse_labels(labels_text):
    return [l.strip() for l in (labels_text or "").split(",") if l.strip()]


def render_frame(folder_path, image_files, frame_idx, dots_by_frame, preview_masks=None, label_order=None):
    # Load a frame, resize to display resolution, draw any preview masks and dots, return PIL image
    # Also returns original and display dimensions needed for coordinate conversion
    img_path = os.path.join(folder_path, image_files[frame_idx])
    orig_img = Image.open(img_path).convert("RGB")
    display_img, orig_w, orig_h, disp_w, disp_h = resize_to_display(orig_img)

    if preview_masks and frame_idx in preview_masks:
        display_img = overlay_masks(display_img, preview_masks[frame_idx], label_order=label_order)

    scale_x = disp_w / orig_w
    scale_y = disp_h / orig_h

    draw = ImageDraw.Draw(display_img)
    for (ox, oy, dot_type, label) in dots_by_frame.get(frame_idx, []):
        dx = int(ox * scale_x)
        dy = int(oy * scale_y)
        color = "green" if dot_type == 1 else "red"
        draw.ellipse(
            [dx - DOT_RADIUS, dy - DOT_RADIUS, dx + DOT_RADIUS, dy + DOT_RADIUS],
            fill=color, outline="white", width=2
        )
        draw.text((dx + DOT_RADIUS + 3, dy - DOT_RADIUS), label, fill=color)

    return display_img, orig_w, orig_h, disp_w, disp_h


# Event handlers

def load_prompts_from_json(json_path):
    dots_by_frame = {}
    coordinates = []
    labels = []
    if not os.path.isfile(json_path):
        return dots_by_frame, coordinates, labels
    with open(json_path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return dots_by_frame, coordinates, labels
    for label, pts in raw.items():
        if label not in labels:
            labels.append(label)
        if not isinstance(pts, list):
            continue
        for pt in pts:
            fidx = int(pt.get("frame", 0))
            ox = int(pt["x"])
            oy = int(pt["y"])
            dot_type = int(pt.get("positive", 1))
            dots_by_frame.setdefault(fidx, []).append((ox, oy, dot_type, label))
            coordinates.append({
                "x": ox,
                "y": oy,
                "positive": dot_type,
                "label": label,
                "frame": fidx,
            })
    return dots_by_frame, coordinates, labels


def load_folder(folder_path, current_labels_text="", output_path=""):
    folder_path = folder_path.strip().strip('"')
    if not folder_path or not os.path.isdir(folder_path):
        return None, "", [], gr.update(maximum=1, value=0), "Invalid folder path.", {}, [], {}, gr.update(), gr.update()

    image_files = get_image_files(folder_path)
    if not image_files:
        return None, "", [], gr.update(maximum=1, value=0), "No image files found in folder.", {}, [], {}, gr.update(), gr.update()

    dots_by_frame = {}
    coordinates = []
    preview_masks = {}
    labels_text_update = gr.update()
    dropdown_update = gr.update()
    prompts_note = ""

    output_path_clean = (output_path or "").strip().strip('"')
    candidate_jsons = [p for p in (output_path_clean, os.path.join(folder_path, "prompts.json")) if p and os.path.isfile(p)]
    if candidate_jsons:
        prompt_file = candidate_jsons[0]
        dots_by_frame, coordinates, loaded_labels = load_prompts_from_json(prompt_file)
        if loaded_labels:
            existing = _parse_labels(current_labels_text)
            merged = list(dict.fromkeys(loaded_labels + existing))
            labels_text_update = ",".join(merged)
            dropdown_update = gr.update(choices=merged, value=loaded_labels[0])
            prompts_note = f" | Loaded {len(coordinates)} prompts from {os.path.basename(prompt_file)}"

    label_order = _parse_labels(labels_text_update if isinstance(labels_text_update, str) else current_labels_text)
    display_img, _, _, _, _ = render_frame(folder_path, image_files, 0, dots_by_frame, preview_masks, label_order=label_order)

    slider_update = gr.update(minimum=0, maximum=max(1, len(image_files) - 1), value=0, step=1)
    status = f"Loaded {len(image_files)} frames.  Frame 1 / {len(image_files)}{prompts_note}"

    return display_img, folder_path, image_files, slider_update, status, dots_by_frame, coordinates, preview_masks, labels_text_update, dropdown_update


def change_frame(frame_idx, folder_path, image_files, dots_by_frame, preview_masks, labels_text=""):
    if not image_files:
        return None, "No folder loaded."

    frame_idx = int(frame_idx)
    display_img, _, _, _, _ = render_frame(
        folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=_parse_labels(labels_text)
    )
    status = f"Frame {frame_idx + 1} / {len(image_files)}: {image_files[frame_idx]}"
    return display_img, status


def on_image_click(evt: gr.SelectData, folder_path, image_files, frame_idx,
                   dots_by_frame, coordinates, label, click_mode, preview_masks, labels_text=""):
    if not image_files:
        return None, dots_by_frame, coordinates, "No folder loaded."

    frame_idx = int(frame_idx)

    # click_x, click_y are in display image space (DISPLAY_W x DISPLAY_H region)
    click_x, click_y = evt.index

    # Load original dimensions to compute scale
    img_path = os.path.join(folder_path, image_files[frame_idx])
    orig_img = Image.open(img_path)
    orig_w, orig_h = orig_img.size
    scale = min(DISPLAY_W / orig_w, DISPLAY_H / orig_h)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    # Convert display click to original image pixel coordinates
    orig_x = int(click_x * orig_w / disp_w)
    orig_y = int(click_y * orig_h / disp_h)

    # Clamp to image bounds
    orig_x = max(0, min(orig_w - 1, orig_x))
    orig_y = max(0, min(orig_h - 1, orig_y))

    dot_type = 1 if click_mode == "Positive" else 0

    dots_by_frame = dict(dots_by_frame)  # shallow copy to trigger Gradio state update
    if frame_idx not in dots_by_frame:
        dots_by_frame[frame_idx] = []
    dots_by_frame[frame_idx] = dots_by_frame[frame_idx] + [(orig_x, orig_y, dot_type, label)]

    coordinates = coordinates + [{
        "x": orig_x,
        "y": orig_y,
        "positive": dot_type,
        "label": label,
        "frame": frame_idx,
    }]

    display_img, _, _, _, _ = render_frame(
        folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=_parse_labels(labels_text)
    )

    frame_dot_count = len(dots_by_frame[frame_idx])
    status = (
        f"Frame {frame_idx + 1} / {len(image_files)} | "
        f"Frame dots: {frame_dot_count} | Total dots: {len(coordinates)} | "
        f"Last click → orig pixel ({orig_x}, {orig_y})"
    )
    return display_img, dots_by_frame, coordinates, status


def undo_last(folder_path, image_files, frame_idx, dots_by_frame, coordinates, preview_masks, labels_text=""):
    if not image_files:
        return None, dots_by_frame, coordinates, preview_masks, "No folder loaded."

    frame_idx = int(frame_idx)
    frame_dots = dots_by_frame.get(frame_idx, [])
    label_order = _parse_labels(labels_text)

    if not frame_dots:
        return (
            render_frame(folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=label_order)[0],
            dots_by_frame, coordinates, preview_masks,
            "No dots on this frame to undo."
        )

    dots_by_frame = dict(dots_by_frame)
    dots_by_frame[frame_idx] = frame_dots[:-1]

    preview_masks = dict(preview_masks or {})
    if not dots_by_frame[frame_idx]:
        preview_masks.pop(frame_idx, None)

    # Remove the last coordinate entry that belongs to this frame
    coordinates = list(coordinates)
    for i in range(len(coordinates) - 1, -1, -1):
        if coordinates[i]['frame'] == frame_idx:
            coordinates.pop(i)
            break

    display_img, _, _, _, _ = render_frame(
        folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=label_order
    )
    status = f"Undid last point on frame {frame_idx + 1}.  Total dots: {len(coordinates)}"
    return display_img, dots_by_frame, coordinates, preview_masks, status


def preview_frame_masks(folder_path, image_files, frame_idx, dots_by_frame, preview_masks, model_id, labels_text=""):
    if not image_files:
        return None, preview_masks, "No folder loaded."

    frame_idx = int(frame_idx)
    label_order = _parse_labels(labels_text)
    frame_dots = dots_by_frame.get(frame_idx, [])
    if not frame_dots:
        display_img, _, _, _, _ = render_frame(
            folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=label_order
        )
        return display_img, preview_masks, f"No dots on frame {frame_idx + 1} to preview."

    by_label = {}
    for (ox, oy, dot_type, label) in frame_dots:
        by_label.setdefault(label, []).append((ox, oy, dot_type))

    try:
        import torch
        t0 = time.perf_counter()
        img_path = os.path.join(folder_path, image_files[frame_idx])
        orig_np = np.array(Image.open(img_path).convert("RGB"))
        predictor = get_image_predictor(model_id)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if predictor.device.type == "cuda"
            else nullcontext()
        )
        frame_masks = {}
        score_parts = []
        with autocast_ctx:
            predictor.set_image(orig_np)
            for label, pts in by_label.items():
                coords_arr = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
                labels_arr = np.array([p[2] for p in pts], dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=coords_arr,
                    point_labels=labels_arr,
                    multimask_output=False,
                )
                mask_bool = masks[0].astype(bool)
                frame_masks[label] = mask_bool
                score_parts.append(f"{label}: {int(mask_bool.sum())} px (score {float(scores[0]):.3f})")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        preview_masks = dict(preview_masks or {})
        preview_masks[frame_idx] = frame_masks

        display_img, _, _, _, _ = render_frame(
            folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=label_order
        )
        status = f"Previewed frame {frame_idx + 1} in {elapsed_ms:.0f} ms | " + " | ".join(score_parts)
        return display_img, preview_masks, status
    except ImportError as e:
        display_img, _, _, _, _ = render_frame(
            folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=label_order
        )
        return (
            display_img,
            preview_masks,
            f"Preview requires PyTorch and SAM 2 (run inside Docker or install torch + sam2): {e}",
        )
    except Exception as e:
        display_img, _, _, _, _ = render_frame(
            folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=label_order
        )
        return display_img, preview_masks, f"Preview failed: {e}"


def clear_preview_masks(folder_path, image_files, frame_idx, dots_by_frame, preview_masks, labels_text=""):
    if not image_files:
        return None, {}, "No folder loaded."
    frame_idx = int(frame_idx)
    preview_masks = dict(preview_masks or {})
    preview_masks.pop(frame_idx, None)
    display_img, _, _, _, _ = render_frame(
        folder_path, image_files, frame_idx, dots_by_frame, preview_masks, label_order=_parse_labels(labels_text)
    )
    return display_img, preview_masks, f"Cleared mask preview on frame {frame_idx + 1}."


def save_coordinates(coordinates, labels_text, output_path):
    output_path = output_path.strip().strip('"')
    if not coordinates:
        return "No coordinates to save."
    if not output_path:
        return "Please enter an output path."

    labels = [l.strip() for l in labels_text.split(',') if l.strip()]
    data_to_save = {}
    for label in labels:
        label_coords = [c for c in coordinates if c['label'] == label]
        if label_coords:
            data_to_save[label] = label_coords

    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        return f"Saved {len(coordinates)} coordinates to {output_path}"
    except Exception as e:
        return f"Error saving: {e}"


def update_labels(labels_text):
    labels = [l.strip() for l in labels_text.split(',') if l.strip()]
    if not labels:
        return gr.update()
    return gr.update(choices=labels, value=labels[0])


# UI

def build_app(config_path=None, initial_frames_dir=""):
    cfg = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    default_labels = cfg.get("labels", ["Mother", "Child", "Judge_1", "Judge_2", "Door"])
    default_labels_str = ",".join(default_labels)
    docker_dir = cfg.get("docker_data_dir", "")
    base_data_dir = docker_dir if (docker_dir and os.path.isdir(docker_dir)) else cfg.get("local_data_dir", docker_dir)
    default_frames_dir = initial_frames_dir or (os.path.join(base_data_dir, cfg["frames_dir"]) if base_data_dir and "frames_dir" in cfg else "")
    default_output_path = os.path.join(base_data_dir, cfg["prompt_coordinates"]) if base_data_dir and "prompt_coordinates" in cfg else ""

    with gr.Blocks(title="SAM2 Prompter") as app:
        gr.Markdown("# MCD SAM2 Prompter")

        folder_path_state = gr.State("")
        image_files_state = gr.State([])
        dots_by_frame_state = gr.State({})
        coordinates_state = gr.State([])
        preview_masks_state = gr.State({})

        with gr.Row():
            folder_input = gr.Textbox(
                label="Frames Folder Path",
                placeholder="/path/to/extracted/frames",
                value=default_frames_dir,
                scale=4
            )
            load_btn = gr.Button("Load Folder", variant="primary", scale=1)

        with gr.Row():
            labels_input = gr.Textbox(
                label="ROI Labels (comma-separated)",
                value=default_labels_str,
                scale=4
            )
            update_labels_btn = gr.Button("Update Labels", scale=1)

        frame_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, label="Frame")

        with gr.Row():
            label_dropdown = gr.Dropdown(
                choices=default_labels,
                value=default_labels[0],
                label="ROI Label",
                scale=2
            )
            click_mode = gr.Radio(
                ["Positive", "Negative"],
                value="Positive",
                label="Click Mode",
                scale=2
            )
            undo_btn = gr.Button("Undo Last Point", scale=1)

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=SAM2_MODEL_CHOICES,
                value=SAM2_MODEL_CHOICES[0],
                label="SAM 2 Preview Model",
                scale=3
            )
            preview_btn = gr.Button("Preview Mask", variant="secondary", scale=1)
            clear_preview_btn = gr.Button("Clear Preview", scale=1)

        # Height is fixed to DISPLAY_H so the browser renders at a known size,
        # keeping the coordinate conversion accurate.
        image_display = gr.Image(
            label="Frame  (click to annotate)",
            type="pil",
            interactive=True,
            height=DISPLAY_H,
        )

        status_label = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            output_path_input = gr.Textbox(
                label="Output JSON Path",
                placeholder="/path/to/output_coordinates.json",
                value=default_output_path,
                scale=4
            )
            save_btn = gr.Button("Save Coordinates", variant="primary", scale=1)

        load_outputs = [
            image_display, folder_path_state, image_files_state, frame_slider,
            status_label, dots_by_frame_state, coordinates_state, preview_masks_state,
            labels_input, label_dropdown,
        ]
        load_btn.click(
            load_folder,
            inputs=[folder_input, labels_input, output_path_input],
            outputs=load_outputs,
        )

        if default_frames_dir:
            app.load(
                load_folder,
                inputs=[folder_input, labels_input, output_path_input],
                outputs=load_outputs,
            )

        frame_slider.change(
            change_frame,
            inputs=[frame_slider, folder_path_state, image_files_state, dots_by_frame_state, preview_masks_state, labels_input],
            outputs=[image_display, status_label]
        )

        image_display.select(
            on_image_click,
            inputs=[folder_path_state, image_files_state, frame_slider, dots_by_frame_state,
                    coordinates_state, label_dropdown, click_mode, preview_masks_state, labels_input],
            outputs=[image_display, dots_by_frame_state, coordinates_state, status_label]
        )

        undo_btn.click(
            undo_last,
            inputs=[folder_path_state, image_files_state, frame_slider, dots_by_frame_state, coordinates_state, preview_masks_state, labels_input],
            outputs=[image_display, dots_by_frame_state, coordinates_state, preview_masks_state, status_label]
        )

        preview_btn.click(
            preview_frame_masks,
            inputs=[folder_path_state, image_files_state, frame_slider, dots_by_frame_state, preview_masks_state, model_dropdown, labels_input],
            outputs=[image_display, preview_masks_state, status_label]
        )

        clear_preview_btn.click(
            clear_preview_masks,
            inputs=[folder_path_state, image_files_state, frame_slider, dots_by_frame_state, preview_masks_state, labels_input],
            outputs=[image_display, preview_masks_state, status_label]
        )

        save_btn.click(
            save_coordinates,
            inputs=[coordinates_state, labels_input, output_path_input],
            outputs=[status_label]
        )

        update_labels_btn.click(
            update_labels,
            inputs=[labels_input],
            outputs=[label_dropdown]
        )

    return app


if __name__ == "__main__":
    default_cfg = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    )
    parser = argparse.ArgumentParser(description="MCD SAM2 Prompter")
    parser.add_argument(
        "--config",
        default="/config/config.yaml" if os.path.isfile("/config/config.yaml") else default_cfg,
        help="Path to config.yaml",
    )
    parser.add_argument("--frames-dir", default="", help="Optional default frames folder path")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    app = build_app(args.config, initial_frames_dir=args.frames_dir)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        css=".gradio-container { max-width: 1000px !important; margin: 0 auto !important; }",
    )
