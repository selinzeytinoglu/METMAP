import gradio as gr
from PIL import Image
import json, os

DISPLAY_W, DISPLAY_H = 1280, 720

def get_projects():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "projects.json")
    if os.path.exists(path):
        with open(path, 'r') as f: return json.load(f).get('projects', [])
    return []

def get_image_files(folder_path):
    if not os.path.isdir(folder_path): return []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    # Sort files naturally
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.splitext(f)[0])) or 0))
    return files

def load_folder(folder_path):
    files = get_image_files(folder_path)
    if not files: return None, f"No image files found in {folder_path}.", gr.update(minimum=0, maximum=1, value=0, interactive=False)
    
    img = Image.open(os.path.join(folder_path, files[0])).convert("RGB")
    scale = min(DISPLAY_W / img.size[0], DISPLAY_H / img.size[1])
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    
    max_frame = max(1, len(files)-1)
    return img, f"Loaded {len(files)} images from {folder_path}.", gr.update(minimum=0, maximum=max_frame, value=0, interactive=True)

def build_app():
    projects = get_projects()
    
    default_project = projects[0]['name'] if projects else None
    default_folders = [(k, v) for k, v in projects[0]['folders'].items()] if projects else []
    default_folder_val = default_folders[0][1] if default_folders else None

    with gr.Blocks(title="SAM2 Prompter") as app:
        gr.Markdown("# MCD SAM2 Prompter")
        with gr.Row():
            p_dd = gr.Dropdown(label="Project", choices=[p['name'] for p in projects], value=default_project, interactive=True)
            f_dd = gr.Dropdown(label="Folder (Exact Path)", choices=default_folders, value=default_folder_val, interactive=True)
            load_btn = gr.Button("Load", variant="primary")
        
        # Link project to folder dropdown
        def update_folders(project_name):
            project = next((p for p in projects if p['name'] == project_name), None)
            if project:
                choices = [(k, v) for k, v in project['folders'].items()]
                val = choices[0][1] if choices else None
                return gr.update(choices=choices, value=val)
            return gr.update(choices=[], value=None)

        p_dd.change(update_folders, inputs=[p_dd], outputs=[f_dd])
        
        frame_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Frame", value=0)
        img_display = gr.Image(type="pil", height=DISPLAY_H)
        status = gr.Textbox(label="Status")

        # f_dd directly holds the exact file path now
        load_btn.click(load_folder, inputs=[f_dd], outputs=[img_display, status, frame_slider])
    return app

if __name__ == "__main__":
    build_app().launch()
