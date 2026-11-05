import gradio as gr
from PIL import Image, ImageDraw
import json
import os
import yaml

# Fixed display resolution. The gr.Image component height is set to match
# DISPLAY_H so the browser renders the image at exactly this size.
DISPLAY_W = 1280
DISPLAY_H = 720
DOT_RADIUS = 4


# Helpers

def get_image_files(folder_path):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else f)
    return files


def resize_to_display(orig_img):
    # Resize a PIL image to fit within DISPLAY_W x DISPLAY_H maintaining aspect ratio
    orig_w, orig_h = orig_img.size
    scale = min(DISPLAY_W / orig_w, DISPLAY_H / orig_h)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    return orig_img.resize((disp_w, disp_h), Image.LANCZOS), orig_w, orig_h, disp_w, disp_h


def render_frame(folder_path, image_files, frame_idx, dots_by_frame):
    # Load a frame, resize to display resolution, draw any existing dots, return PIL image
    # Also returns original and display dimensions needed for coordinate conversion
    img_path = os.path.join(folder_path, image_files[frame_idx])
    orig_img = Image.open(img_path).convert("RGB")
    display_img, orig_w, orig_h, disp_w, disp_h = resize_to_display(orig_img)

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

def load_folder(folder_path):
    folder_path = folder_path.strip().strip('"')
    if not folder_path or not os.path.isdir(folder_path):
        return None, "", [], gr.update(maximum=0, value=0), "Invalid folder path.", {}

    image_files = get_image_files(folder_path)
    if not image_files:
        return None, "", [], gr.update(maximum=0, value=0), "No image files found in folder.", {}

    dots_by_frame = {}
    display_img, _, _, _, _ = render_frame(folder_path, image_files, 0, dots_by_frame)

    slider_update = gr.update(minimum=0, maximum=len(image_files) - 1, value=0, step=1)
    status = f"Loaded {len(image_files)} frames.  Frame 1 / {len(image_files)}"

    return display_img, folder_path, image_files, slider_update, status, dots_by_frame


def change_frame(frame_idx, folder_path, image_files, dots_by_frame):
    if not image_files:
        return None, "No folder loaded."

    frame_idx = int(frame_idx)
    display_img, _, _, _, _ = render_frame(folder_path, image_files, frame_idx, dots_by_frame)
    status = f"Frame {frame_idx + 1} / {len(image_files)}: {image_files[frame_idx]}"
    return display_img, status


def on_image_click(evt: gr.SelectData, folder_path, image_files, frame_idx,
                   dots_by_frame, coordinates, label, click_mode):
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

    display_img, _, _, _, _ = render_frame(folder_path, image_files, frame_idx, dots_by_frame)

    frame_dot_count = len(dots_by_frame[frame_idx])
    status = (
        f"Frame {frame_idx + 1} / {len(image_files)} | "
        f"Frame dots: {frame_dot_count} | Total dots: {len(coordinates)} | "
        f"Last click → orig pixel ({orig_x}, {orig_y})"
    )
    return display_img, dots_by_frame, coordinates, status


def undo_last(folder_path, image_files, frame_idx, dots_by_frame, coordinates):
    if not image_files:
        return None, dots_by_frame, coordinates, "No folder loaded."

    frame_idx = int(frame_idx)
    frame_dots = dots_by_frame.get(frame_idx, [])

    if not frame_dots:
        return (
            render_frame(folder_path, image_files, frame_idx, dots_by_frame)[0],
            dots_by_frame, coordinates,
            "No dots on this frame to undo."
        )

    dots_by_frame = dict(dots_by_frame)
    dots_by_frame[frame_idx] = frame_dots[:-1]

    # Remove the last coordinate entry that belongs to this frame
    coordinates = list(coordinates)
    for i in range(len(coordinates) - 1, -1, -1):
        if coordinates[i]['frame'] == frame_idx:
            coordinates.pop(i)
            break

    display_img, _, _, _, _ = render_frame(folder_path, image_files, frame_idx, dots_by_frame)
    status = f"Undid last point on frame {frame_idx + 1}.  Total dots: {len(coordinates)}"
    return display_img, dots_by_frame, coordinates, status


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

def build_app(config_path=None):
    cfg = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    default_labels = cfg.get("labels", ["Mother", "Child", "Judge_1", "Judge_2", "Door"])
    default_labels_str = ",".join(default_labels)
    docker_data_dir = cfg.get("docker_data_dir", "")
    default_frames_dir = os.path.join(docker_data_dir, cfg["frames_dir"]) if docker_data_dir and "frames_dir" in cfg else ""
    default_output_path = os.path.join(docker_data_dir, cfg["prompt_coordinates"]) if docker_data_dir and "prompt_coordinates" in cfg else ""

    with gr.Blocks(title="SAM2 Prompter") as app:
        gr.Markdown("# MCD SAM2 Prompter")

        folder_path_state = gr.State("")
        image_files_state = gr.State([])
        dots_by_frame_state = gr.State({})
        coordinates_state = gr.State([])

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

        frame_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Frame")

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

        load_btn.click(
            load_folder,
            inputs=[folder_input],
            outputs=[image_display, folder_path_state, image_files_state, frame_slider, status_label, dots_by_frame_state]
        )

        if default_frames_dir:
            app.load(
                load_folder,
                inputs=[folder_input],
                outputs=[image_display, folder_path_state, image_files_state, frame_slider, status_label, dots_by_frame_state]
            )

        frame_slider.change(
            change_frame,
            inputs=[frame_slider, folder_path_state, image_files_state, dots_by_frame_state],
            outputs=[image_display, status_label]
        )

        image_display.select(
            on_image_click,
            inputs=[folder_path_state, image_files_state, frame_slider, dots_by_frame_state,
                    coordinates_state, label_dropdown, click_mode],
            outputs=[image_display, dots_by_frame_state, coordinates_state, status_label]
        )

        undo_btn.click(
            undo_last,
            inputs=[folder_path_state, image_files_state, frame_slider, dots_by_frame_state, coordinates_state],
            outputs=[image_display, dots_by_frame_state, coordinates_state, status_label]
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
    config_path = "/config/config.yaml"
    app = build_app(config_path)
    app.launch(server_name="0.0.0.0", css=".gradio-container { max-width: 1000px !important; margin: 0 auto !important; }")
