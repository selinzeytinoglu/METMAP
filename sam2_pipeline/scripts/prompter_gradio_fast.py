#!/usr/bin/env python3
"""
prompter_gradio_fast.py - Interactive ROI annotation tool using Gradio

This tool provides a web-based UI for annotating regions of interest in video frames.
Users select a project/participant, load frames, click to annotate ROIs, and save coordinates.

Usage:
    python3 prompter_gradio.py
    Then open: http://127.0.0.1:7860
"""

import json
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings

# Suppress a known Gradio/Starlette compatibility warning emitted by the queue route.
try:
    from starlette.exceptions import StarletteDeprecationWarning
except ImportError:
    StarletteDeprecationWarning = Warning

warnings.filterwarnings(
    "ignore",
    message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated\. Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead\.",
    category=StarletteDeprecationWarning,
    module=r"gradio\.routes",
)

# pyrefly: ignore [missing-import]
import gradio as gr
from PIL import Image, ImageDraw


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config() -> Dict:
    """Load configuration from config.yaml."""
    import os
    import yaml
    
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths relative to the folder containing config.yaml
    config_dir = Path(__file__).parent
    data_root = Path(os.path.normpath(config_dir / config["data_root"])).absolute()
    config["_data_root_resolved"] = data_root
    config["_frames_root"] = data_root / config["frames_subdir"]
    config["_prompts_root"] = data_root / config["prompts_subdir"]
    
    return config


# Global config
CONFIG = load_config()
DISPLAY_W = CONFIG.get("display_width", 1280)
DISPLAY_H = CONFIG.get("display_height", 720)
DOT_RADIUS = CONFIG.get("dot_radius", 4)
BOX_HANDLE_RADIUS = int(CONFIG.get("box_handle_radius", max(8, DOT_RADIUS + 5)))
DEFAULT_LABELS = CONFIG.get("default_labels", ["Mother", "Child", "Judge_1", "Judge_2", "Door"])

KEYBOARD_NAV_JS = r"""
() => {
  if (window.__sam2PrompterKeyboardNavInstalled) return;
  window.__sam2PrompterKeyboardNavInstalled = true;

  const ignoredTags = new Set(["INPUT", "TEXTAREA", "SELECT", "BUTTON"]);

  function setNativeValue(element, value) {
    if (!element) return;
    const descriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(element), "value");
    if (descriptor && descriptor.set) {
      descriptor.set.call(element, value);
    } else {
      element.value = value;
    }
  }

  function stepFrame(delta) {
    const root = document.getElementById("frame-slider");
    if (!root) return false;

    const rangeInput = root.querySelector('input[type="range"]');
    const numberInput = root.querySelector('input[type="number"]');
    const primaryInput = rangeInput || numberInput;
    if (!primaryInput || primaryInput.disabled) return false;

    const min = Number(primaryInput.min || 0);
    const max = Number(primaryInput.max || 0);
    const current = Number(primaryInput.value || 0);
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min || !Number.isFinite(current)) {
      return false;
    }

    const next = Math.max(min, Math.min(max, current + delta));
    if (next === current) return true;

    setNativeValue(rangeInput, String(next));
    setNativeValue(numberInput, String(next));
    primaryInput.dispatchEvent(new Event("input", { bubbles: true }));
    primaryInput.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  function disableImageDragging() {
    document.querySelectorAll("img").forEach((img) => {
      if (img.dataset.sam2NoDrag === "1") return;
      img.dataset.sam2NoDrag = "1";
      img.draggable = false;
      img.style.userSelect = "none";
      img.style.webkitUserDrag = "none";
      img.addEventListener("dragstart", (event) => event.preventDefault());
    });
  }

  disableImageDragging();
  const observer = new MutationObserver(disableImageDragging);
  observer.observe(document.body, { childList: true, subtree: true });

  document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented) return;
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;

    const target = event.target;
    const insideFrameSlider = target && target.closest && target.closest("#frame-slider");
    if (target && !insideFrameSlider && (target.isContentEditable || ignoredTags.has(target.tagName))) return;

    const delta = event.key === "ArrowRight" ? 1 : -1;
    if (stepFrame(delta)) event.preventDefault();
  }, true);
}
"""

# Bounded display-frame cache. These are resized RGB frames, not full-resolution
# originals, so nearby navigation can be warmed without loading the whole video.
FRAME_CACHE_SIZE = int(CONFIG.get("frame_cache_size", 192))
PREFETCH_AHEAD = int(CONFIG.get("prefetch_ahead", 48))
PREFETCH_BEHIND = int(CONFIG.get("prefetch_behind", 8))
_FRAME_CACHE = OrderedDict()
_FRAME_CACHE_LOCK = threading.RLock()
_PREFETCH_CONDITION = threading.Condition()
_PREFETCH_REQUEST = None
_PREFETCH_WORKER_STARTED = False


# ============================================================================
# Projects Management
# ============================================================================

def get_projects() -> Dict:
    """
    Get projects configuration, auto-generating if projects.json doesn't exist.
    Returns: Dict with structure {"projects": [...]}
    """
    projects_path = Path(__file__).parent / "projects.json"
    
    if not projects_path.exists():
        print("projects.json not found. Generating...")
        try:
            subprocess.run(
                [sys.executable, str(Path(__file__).parent / "discover_frames.py")],
                check=True,
                capture_output=True
            )
        except Exception as e:
            print(f"Error generating projects.json: {e}")
            return {"projects": []}
    
    try:
        with open(projects_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading projects.json: {e}")
        return {"projects": []}


def get_project_names() -> List[str]:
    """Get list of project (dataset) names."""
    projects = get_projects()
    return [p["name"] for p in projects["projects"]]


def get_participant_ids(project_name: str) -> List[str]:
    """Get list of participant IDs for a project."""
    projects = get_projects()
    for project in projects["projects"]:
        if project["name"] == project_name:
            return sorted(project["data"].keys())
    return []


def get_frames_path(project_name: str, participant_id: str) -> Optional[Path]:
    """Get the frames path for a participant."""
    projects = get_projects()
    for project in projects["projects"]:
        if project["name"] == project_name:
            if participant_id in project["data"]:
                frames_rel = project["data"][participant_id]["frames"]
                return CONFIG["_data_root_resolved"] / frames_rel
    return None


def _numeric_sort_key(filename: str):
    """Sort key that handles numeric filenames (e.g. '1.jpg' before '10.jpg')."""
    import os
    stem = os.path.splitext(filename)[0]
    try:
        return (0, int(stem), filename)
    except ValueError:
        return (1, 0, filename)


def get_frames_list(frames_path: Optional[Path]) -> List[str]:
    """Get sorted list of frame files from frames_path (numeric sort for pipeline compat)."""
    if not frames_path or not frames_path.exists():
        return []
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    try:
        import os
        files = []
        with os.scandir(frames_path) as it:
            for entry in it:
                if entry.is_file():
                    name = entry.name
                    _, ext = os.path.splitext(name)
                    if ext.lower() in valid_exts:
                        files.append(name)
        files.sort(key=_numeric_sort_key)
        return files
    except Exception:
        return []


def get_recommended_output_path(project_name: str, participant_id: str) -> Tuple[Optional[str], Optional[Path]]:
    """
    Generate recommended output (prompts) path for saving coordinates.
    Returns: (path_as_string, directory_path) or (None, None) if can't generate
    """
    projects = get_projects()
    for project in projects["projects"]:
        if project["name"] == project_name:
            if participant_id in project["data"]:
                prompts_rel = project["data"][participant_id]["prompts"]
                prompts_path = CONFIG["_data_root_resolved"] / prompts_rel
                return (str(prompts_path), prompts_path)
    return (None, None)


# ============================================================================
# Frame Loading and Annotation
# ============================================================================


def resize_to_display(orig_img: Image.Image) -> Tuple[Image.Image, int, int, int, int]:
    """Resize a PIL image to fit within DISPLAY_W x DISPLAY_H maintaining aspect ratio."""
    orig_w, orig_h = orig_img.size
    scale = min(DISPLAY_W / orig_w, DISPLAY_H / orig_h)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    return orig_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS), orig_w, orig_h, disp_w, disp_h


def _frame_cache_key(frames_path: Path, frame_name: str) -> Tuple[str, str]:
    return (str(frames_path), frame_name)


def _evict_frame_cache_locked() -> None:
    while len(_FRAME_CACHE) > max(1, FRAME_CACHE_SIZE):
        _FRAME_CACHE.popitem(last=False)


def is_display_frame_cached(frames_path: Path, frame_name: str) -> bool:
    key = _frame_cache_key(frames_path, frame_name)
    with _FRAME_CACHE_LOCK:
        return key in _FRAME_CACHE


def get_cached_display_frame(
    frames_path: Path,
    frame_name: str,
    copy_image: bool = True,
) -> Optional[Tuple[Image.Image, int, int, int, int, bool]]:
    """
    Return a resized display frame and dimensions.

    The cached object is the resized base frame before annotation. Callers that draw
    should use the returned copy so the cache stays immutable.
    """
    key = _frame_cache_key(frames_path, frame_name)
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.pop(key, None)
        if cached is not None:
            _FRAME_CACHE[key] = cached
            display_img, orig_w, orig_h, disp_w, disp_h = cached
            if copy_image:
                display_img = display_img.copy()
            return display_img, orig_w, orig_h, disp_w, disp_h, True

    try:
        with Image.open(frames_path / frame_name) as orig_img:
            rgb_img = orig_img.convert("RGB")
        display_img, orig_w, orig_h, disp_w, disp_h = resize_to_display(rgb_img)
    except Exception as exc:
        print(f"Error loading frame {frames_path / frame_name}: {exc}")
        return None

    cached = (display_img, orig_w, orig_h, disp_w, disp_h)
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[key] = cached
        _evict_frame_cache_locked()

    output_img = display_img.copy() if copy_image else display_img
    return output_img, orig_w, orig_h, disp_w, disp_h, False


def _prefetch_indices(center_idx: int, frame_count: int) -> List[int]:
    forward = range(center_idx + 1, min(frame_count, center_idx + 1 + PREFETCH_AHEAD))
    backward = range(max(0, center_idx - PREFETCH_BEHIND), center_idx)
    return list(forward) + list(reversed(list(backward)))


def schedule_prefetch(frames_path: Path, frames_list: List[str], center_idx: int) -> None:
    """Ask the single background worker to warm frames near center_idx."""
    if not frames_list or (PREFETCH_AHEAD <= 0 and PREFETCH_BEHIND <= 0):
        return

    global _PREFETCH_REQUEST, _PREFETCH_WORKER_STARTED
    request = (str(frames_path), list(frames_list), int(center_idx))
    with _PREFETCH_CONDITION:
        _PREFETCH_REQUEST = request
        if not _PREFETCH_WORKER_STARTED:
            worker = threading.Thread(target=_prefetch_loop, daemon=True)
            worker.start()
            _PREFETCH_WORKER_STARTED = True
        _PREFETCH_CONDITION.notify()


def _prefetch_loop() -> None:
    global _PREFETCH_REQUEST
    while True:
        with _PREFETCH_CONDITION:
            while _PREFETCH_REQUEST is None:
                _PREFETCH_CONDITION.wait()
            request = _PREFETCH_REQUEST
            _PREFETCH_REQUEST = None

        frames_path_str, frames_list, center_idx = request
        frames_path = Path(frames_path_str)
        for idx in _prefetch_indices(center_idx, len(frames_list)):
            with _PREFETCH_CONDITION:
                if _PREFETCH_REQUEST is not None:
                    break
            frame_name = frames_list[idx]
            if not is_display_frame_cached(frames_path, frame_name):
                get_cached_display_frame(frames_path, frame_name, copy_image=False)


def load_frame(frames_path: Path, frame_name: str) -> Optional[Image.Image]:
    """Load a full-resolution frame image for rare callers that need it."""
    try:
        with Image.open(frames_path / frame_name) as img:
            return img.convert("RGB")
    except Exception:
        return None


def coerce_frames_list(frames_list) -> List[str]:
    if isinstance(frames_list, str):
        try:
            frames_list = json.loads(frames_list)
        except Exception:
            return []
    return frames_list if isinstance(frames_list, list) else []


def coerce_frame_index(frame_value, frames_list: List[str]) -> Optional[int]:
    if frame_value is None:
        return None

    if isinstance(frame_value, bool):
        return None

    if isinstance(frame_value, int):
        idx = frame_value
    elif isinstance(frame_value, float) and frame_value.is_integer():
        idx = int(frame_value)
    elif isinstance(frame_value, str):
        value = frame_value.strip()
        if value.isdigit():
            idx = int(value)
        else:
            frame_to_idx = {name: i for i, name in enumerate(frames_list)}
            idx = frame_to_idx.get(value)
            if idx is None:
                return None
    else:
        return None

    if 0 <= idx < len(frames_list):
        return idx
    return None


def _coerce_positive(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "negative", "no"}
    return bool(value)


def normalise_box(box) -> Optional[List[int]]:
    """Return a sorted [x1, y1, x2, y2] box or None when invalid."""
    if isinstance(box, dict):
        key_sets = (
            ("x1", "y1", "x2", "y2"),
            ("x_min", "y_min", "x_max", "y_max"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        )
        for keys in key_sets:
            if all(key in box for key in keys):
                box = [box[key] for key in keys]
                break
        else:
            return None

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None

    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def is_box_annotation(coord: Dict) -> bool:
    if not isinstance(coord, dict):
        return False
    box_keys = {"box", "bbox", "x1", "x_min", "xmin", "left"}
    return coord.get("type") == "box" or any(key in coord for key in box_keys)


def box_corners(box) -> List[Tuple[int, int]]:
    box = normalise_box(box)
    if box is None:
        return []
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def find_box_for_label(coords: List[Dict], label: str) -> Tuple[Optional[Dict], Optional[List[int]]]:
    for coord in coords or []:
        if is_box_annotation(coord) and coord.get("label", "") == label:
            box = normalise_box(coord.get("box", coord.get("bbox", coord)))
            if box is not None:
                return coord, box
    return None, None


def remove_box_for_label(coords: List[Dict], label: str) -> List[Dict]:
    return [
        coord for coord in coords or []
        if not (is_box_annotation(coord) and coord.get("label", "") == label)
    ]


def nearest_box_corner(box, click_x: float, click_y: float, scale_x: float, scale_y: float) -> Optional[int]:
    best_idx = None
    best_distance = None
    for idx, (x, y) in enumerate(box_corners(box)):
        dx = (x * scale_x) - click_x
        dy = (y * scale_y) - click_y
        distance = (dx * dx + dy * dy) ** 0.5
        if best_distance is None or distance < best_distance:
            best_idx = idx
            best_distance = distance
    if best_distance is not None and best_distance <= BOX_HANDLE_RADIUS * 2:
        return best_idx
    return None


def move_box_corner(box, corner_idx: int, x: int, y: int) -> Optional[List[int]]:
    box = normalise_box(box)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if corner_idx == 0:
        x1, y1 = x, y
    elif corner_idx == 1:
        x2, y1 = x, y
    elif corner_idx == 2:
        x2, y2 = x, y
    elif corner_idx == 3:
        x1, y2 = x, y
    else:
        return None
    return normalise_box([x1, y1, x2, y2])


def box_from_points(points: List[Dict]) -> Optional[List[int]]:
    if len(points) < 2:
        return None
    try:
        xs = [int(point["x"]) for point in points]
        ys = [int(point["y"]) for point in points]
    except (KeyError, TypeError, ValueError):
        return None
    return normalise_box([min(xs), min(ys), max(xs), max(ys)])


def valid_box(box: Optional[List[int]]) -> bool:
    return bool(box and box[0] != box[2] and box[1] != box[3])


def count_prompt_types(coords: List[Dict]) -> Tuple[int, int]:
    points = 0
    boxes = 0
    for coord in coords or []:
        if is_box_annotation(coord):
            boxes += 1
        else:
            points += 1
    return points, boxes


def get_frame_dots(dots_by_frame: Dict, frame_idx: int) -> List[Dict]:
    if not dots_by_frame:
        return []
    if frame_idx in dots_by_frame:
        return dots_by_frame.get(frame_idx, [])
    return dots_by_frame.get(str(frame_idx), [])


def normalise_dots_by_frame(dots_by_frame: Dict) -> Dict[int, List[Dict]]:
    normalised: Dict[int, List[Dict]] = {}
    if not dots_by_frame:
        return normalised

    for frame_key, coords in dots_by_frame.items():
        try:
            frame_idx = int(frame_key)
        except (TypeError, ValueError):
            continue
        normalised[frame_idx] = list(coords or [])
    return normalised


def load_prompt_coordinates(prompt_path: str, frames_list: List[str]) -> Tuple[Dict[int, List[Dict]], str]:
    """Load existing prompt JSON, accepting both integer and filename frame fields."""
    if not prompt_path:
        return {}, "No prompt JSON selected; starting empty."

    prompt_file = Path(prompt_path)
    if not prompt_file.exists():
        return {}, f"No existing prompt JSON at {prompt_file.name}; starting empty."

    try:
        with open(prompt_file, "r") as f:
            data = json.load(f)
    except Exception as exc:
        return {}, f"Could not read existing prompt JSON: {exc}"

    dots_by_frame: Dict[int, List[Dict]] = {}
    loaded = 0
    skipped = 0
    for label, coords in data.items():
        if not isinstance(coords, list):
            skipped += 1
            continue
        for coord in coords:
            if not isinstance(coord, dict):
                skipped += 1
                continue
            frame_idx = coerce_frame_index(coord.get("frame"), frames_list)
            if frame_idx is None:
                skipped += 1
                continue
            label_text = coord.get("label") or label
            if is_box_annotation(coord):
                box = normalise_box(coord.get("box", coord.get("bbox", coord)))
                if box is None:
                    skipped += 1
                    continue
                dot = {
                    "type": "box",
                    "box": box,
                    "label": label_text,
                }
            else:
                try:
                    dot = {
                        "type": "point",
                        "x": int(coord["x"]),
                        "y": int(coord["y"]),
                        "positive": _coerce_positive(coord.get("positive", True)),
                        "label": label_text,
                    }
                except Exception:
                    skipped += 1
                    continue
            dots_by_frame.setdefault(frame_idx, []).append(dot)
            loaded += 1

    skipped_msg = f" Skipped {skipped} invalid prompts." if skipped else ""
    return dots_by_frame, f"Loaded {loaded} existing prompts from {prompt_file.name}.{skipped_msg}"


def canonical_prompt_path(project_name: str, participant_id: str) -> str:
    """Return the canonical prompt JSON path from projects.json/config discovery."""
    if not project_name or not participant_id:
        return ""
    path_text, _ = get_recommended_output_path(project_name, participant_id)
    return path_text or ""


def canonical_prompt_path_update(project_name: str, participant_id: str) -> str:
    return canonical_prompt_path(project_name, participant_id)


def normalize_prompt_path_text(prompt_path: str) -> str:
    """Clean a prompt JSON path and apply the same .json default used on save."""
    if not prompt_path:
        return ""
    path_text = str(prompt_path).strip().strip('"')
    if not path_text:
        return ""
    path_obj = Path(path_text)
    if path_obj.suffix == "":
        path_obj = path_obj.with_suffix(".json")
    return str(path_obj)


def resolve_prompt_path(project_name: str, participant_id: str, manual_frames_path: str, output_path: str, auto_prompt_path: str) -> Tuple[str, str]:
    """Return the prompt path to use plus the current auto-generated path."""
    current_path = normalize_prompt_path_text(output_path)
    previous_auto_path = normalize_prompt_path_text(auto_prompt_path)
    using_manual_frames = bool(manual_frames_path and manual_frames_path.strip())

    if using_manual_frames:
        if current_path and current_path != previous_auto_path:
            return current_path, ""
        return "", ""

    next_auto_path = canonical_prompt_path(project_name, participant_id)
    if not current_path or current_path == previous_auto_path:
        return next_auto_path, next_auto_path
    return current_path, next_auto_path


def draw_annotations(image: Image.Image, coordinates: List[Dict], orig_w: int, orig_h: int, disp_w: int, disp_h: int) -> Image.Image:
    """Draw point prompts, box prompts, and box corner handles."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    scale_x = disp_w / orig_w if orig_w else 1.0
    scale_y = disp_h / orig_h if orig_h else 1.0

    for coord in coordinates:
        label = coord.get("label", "")
        if is_box_annotation(coord):
            box = normalise_box(coord.get("box", coord.get("bbox", coord)))
            if box is None:
                continue
            x1, y1, x2, y2 = box
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)
            color = "cyan"
            draw.rectangle([dx1, dy1, dx2, dy2], outline=color, width=max(2, DOT_RADIUS))
            handle_r = BOX_HANDLE_RADIUS
            for corner_x, corner_y in box_corners(box):
                cx = int(corner_x * scale_x)
                cy = int(corner_y * scale_y)
                draw.rectangle(
                    [cx - handle_r, cy - handle_r, cx + handle_r, cy + handle_r],
                    fill=color,
                    outline="black",
                    width=2,
                )
            if label:
                try:
                    draw.text((dx1 + 4, max(0, dy1 - 16)), f"{label} box", fill=color)
                except Exception:
                    pass
            continue

        try:
            ox, oy = coord["x"], coord["y"]
        except KeyError:
            continue
        dx = int(ox * scale_x)
        dy = int(oy * scale_y)
        positive = _coerce_positive(coord.get("positive", True))
        color = "green" if positive else "red"

        r = DOT_RADIUS
        draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=color, outline="white", width=2)

        if label:
            try:
                draw.text((dx + r + 4, dy - r - 4), label, fill=color)
            except Exception:
                pass

    return img


def draw_pending_box_overlay(image: Image.Image, pending_box, orig_w: int, orig_h: int, disp_w: int, disp_h: int) -> Image.Image:
    if not isinstance(pending_box, dict):
        return image

    img = image.copy()
    draw = ImageDraw.Draw(img)
    scale_x = disp_w / orig_w if orig_w else 1.0
    scale_y = disp_h / orig_h if orig_h else 1.0
    color = "yellow"

    if pending_box.get("mode") == "edit":
        box = normalise_box(pending_box.get("box"))
        try:
            corner_idx = int(pending_box.get("corner"))
        except (TypeError, ValueError):
            corner_idx = -1
        corners = box_corners(box)
        if 0 <= corner_idx < len(corners):
            ox, oy = corners[corner_idx]
            dx = int(ox * scale_x)
            dy = int(oy * scale_y)
            r = BOX_HANDLE_RADIUS + 5
            draw.rectangle([dx - r, dy - r, dx + r, dy + r], outline=color, width=3)
        return img

    points = pending_box.get("points") or []
    valid_points = []
    for point in points:
        try:
            ox = int(point["x"])
            oy = int(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        valid_points.append({"x": ox, "y": oy})
        dx = int(ox * scale_x)
        dy = int(oy * scale_y)
        r = BOX_HANDLE_RADIUS
        draw.rectangle([dx - r, dy - r, dx + r, dy + r], fill=color, outline="black", width=2)
        try:
            draw.text((dx + r + 3, dy - r), str(len(valid_points)), fill=color)
        except Exception:
            pass

    preview_box = box_from_points(valid_points)
    if preview_box is not None:
        x1, y1, x2, y2 = preview_box
        draw.rectangle(
            [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)],
            outline=color,
            width=2,
        )
    return img

def update_coord_count(dots_by_frame, current_frame):
    """Update prompt count display."""
    dots_by_frame = normalise_dots_by_frame(dots_by_frame)
    frame_idx = None
    try:
        frame_idx = int(current_frame)
    except (TypeError, ValueError):
        pass

    current_coords = dots_by_frame.get(frame_idx, []) if frame_idx is not None else []
    current_points, current_boxes = count_prompt_types(current_coords)
    total_points = 0
    total_boxes = 0
    for coords in dots_by_frame.values():
        points, boxes = count_prompt_types(coords)
        total_points += points
        total_boxes += boxes

    current_total = current_points + current_boxes
    total_prompts = total_points + total_boxes
    return (
        f"Current Frame: {current_total} prompts "
        f"({current_points} points, {current_boxes} boxes) | "
        f"Total: {total_prompts} prompts ({total_points} points, {total_boxes} boxes)"
    )


def render_frame_for_ui(frame_idx, frames_list, active_frames_path, dots_by_frame):
    """Render one frame for the UI using the bounded display-frame cache."""
    frames_list = coerce_frames_list(frames_list)
    if not frames_list or not active_frames_path:
        return None, None, "Status: No folder loaded.", "0 prompts"

    try:
        frame_idx = int(frame_idx)
    except (TypeError, ValueError):
        frame_idx = 0
    frame_idx = max(0, min(len(frames_list) - 1, frame_idx))

    frames_path = Path(active_frames_path)
    frame_name = frames_list[frame_idx]
    start = time.perf_counter()
    cached = get_cached_display_frame(frames_path, frame_name)
    if cached is None:
        return None, frame_idx, f"Status: Error - Could not load frame {frame_name}", update_coord_count(dots_by_frame, frame_idx)

    display_img, orig_w, orig_h, disp_w, disp_h, cache_hit = cached
    coords = get_frame_dots(dots_by_frame, frame_idx)
    annotated_frame = draw_annotations(display_img, coords, orig_w, orig_h, disp_w, disp_h)
    schedule_prefetch(frames_path, frames_list, frame_idx)

    elapsed_ms = (time.perf_counter() - start) * 1000
    source = "cache" if cache_hit else "disk"
    status = f"Status: Frame {frame_idx + 1} / {len(frames_list)}: {frame_name} ({source}, {elapsed_ms:.0f} ms)"
    return annotated_frame, frame_idx, status, update_coord_count(dots_by_frame, frame_idx)


# ============================================================================
# Save Coordinates
# ============================================================================

def save_coordinates(
    dots_by_frame: Dict,
    output_path: str,
    project_name: str,
    participant_id: str
) -> str:
    """
    Save coordinates to a JSON file grouped by label.

    Saved frame values are integer frame indices, matching segment_video.py and
    the original prompter convention. Existing filename-based prompt files are
    accepted on load, but new saves use the downstream-compatible integer form.
    """
    if not output_path or output_path.strip() == "":
        recommended_path, _ = get_recommended_output_path(project_name, participant_id)
        if recommended_path:
            output_path = recommended_path
        else:
            return "Status: Error - Could not generate output path"

    try:
        output_path_obj = Path(normalize_prompt_path_text(output_path))
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        normalised = normalise_dots_by_frame(dots_by_frame)
        if not normalised:
            return "Status: No prompts to save."

        data_to_save = {}
        for frame_idx, coords in sorted(normalised.items()):
            for coord in coords:
                label = coord.get("label", "")
                if not label:
                    print(f"Warning: prompt on frame {frame_idx} has no label and will be skipped.")
                    continue

                if is_box_annotation(coord):
                    box = normalise_box(coord.get("box", coord.get("bbox", coord)))
                    if box is None:
                        print(f"Warning: invalid box prompt for {label} on frame {frame_idx} will be skipped.")
                        continue
                    data_to_save.setdefault(label, []).append({
                        "box": box,
                        "label": label,
                        "frame": int(frame_idx),
                    })
                    continue

                try:
                    x = int(coord["x"])
                    y = int(coord["y"])
                except Exception:
                    print(f"Warning: invalid point prompt for {label} on frame {frame_idx} will be skipped.")
                    continue

                data_to_save.setdefault(label, []).append({
                    "x": x,
                    "y": y,
                    "positive": int(_coerce_positive(coord.get("positive", True))),
                    "label": label,
                    "frame": int(frame_idx),
                })

        with open(output_path_obj, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        total_saved = sum(len(coords) for coords in data_to_save.values())
        return f"Status: Saved {total_saved} prompts to {output_path_obj}"
    except Exception as e:
        return f"Status: Error saving: {e}"


# ============================================================================
# Gradio UI
# ============================================================================

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="SAM2 Prompter") as interface:
        gr.Markdown("# SAM2 Prompter: ROI Annotation")
        
        # State variables
        current_frame_state = gr.State(value="")
        dots_by_frame_state = gr.State(value={})
        frames_list_state = gr.State(value=[])
        active_frames_path_state = gr.State(value="")
        auto_prompt_path_state = gr.State(value="")
        is_positive_state = gr.State(value=True)
        is_box_mode_state = gr.State(value=False)
        pending_box_start_state = gr.State(value=None)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Image display on the left (matches CONFIG display height, scale reduced to avoid margins)
                image_display = gr.Image(
                    label="Frame (click to add point or box prompt)",
                    type="pil",
                    interactive=True,
                    height=DISPLAY_H
                )
            
            with gr.Column(scale=1):
                # Control panel on the right (flat single panel, no tabs to save clicks)
                
                # 1. Load Data
                with gr.Group():
                    gr.Markdown("### 1. Load Data")
                    project_choices = get_project_names()
                    default_project = project_choices[0] if project_choices else None
                    default_participants = get_participant_ids(default_project) if default_project else []
                    default_participant = default_participants[0] if default_participants else None

                    project_dropdown = gr.Dropdown(
                        choices=project_choices,
                        value=default_project,
                        label="Project (Dataset)",
                        interactive=True
                    )
                    participant_dropdown = gr.Dropdown(
                        choices=default_participants,
                        value=default_participant,
                        label="Participant ID",
                        interactive=True
                    )
                    manual_frames_input = gr.Textbox(
                        label="Or Enter Manual Folder Path Directly",
                        placeholder="Y:\\...\\sam2_frames\\speech_task\\...",
                        interactive=True
                    )
                    with gr.Row():
                        load_btn = gr.Button("Load Folder / Participant", variant="primary")
                        load_empty_btn = gr.Button("Load Empty (Ignore Prompts)")
                
                # 2. Navigate & Annotate
                with gr.Group():
                    gr.Markdown("### 2. Navigate & Annotate")
                    
                    # Active ROI label selection (Radio)
                    with gr.Row():
                        labels_input = gr.Textbox(
                            label="Edit Labels (comma separated)",
                            value=",".join(DEFAULT_LABELS),
                            scale=3
                        )
                        update_labels_btn = gr.Button("Update", scale=1)

                    active_label_radio = gr.Radio(
                        choices=DEFAULT_LABELS,
                        value=DEFAULT_LABELS[0] if DEFAULT_LABELS else "",
                        label="Select Active ROI Label"
                    )
                    
                    with gr.Row():
                        positive_btn = gr.Button("Add Positive", variant="primary")
                        negative_btn = gr.Button("Add Negative", variant="stop")
                        box_btn = gr.Button("Add/Edit Box")
                    
                    frame_slider = gr.Slider(
                        minimum=0, maximum=1, step=1,
                        label="Frame Number",
                        interactive=False,
                        elem_id="frame-slider"
                    )
                    
                    with gr.Row():
                        prev_btn = gr.Button("Previous")
                        next_btn = gr.Button("Next")
                        undo_btn = gr.Button("Undo Last")
                        
                    coord_count = gr.Textbox(
                        value="0 prompts",
                        label="Prompt Count",
                        interactive=False
                    )
                
                # 3. Save Coordinates & Status
                with gr.Group():
                    gr.Markdown("### 3. Save Coordinates")
                    output_path_input = gr.Textbox(
                        value="",
                        label="Prompt JSON Path (editable; blank uses selection default)",
                        placeholder=canonical_prompt_path(default_project, default_participant) or "Y:\\...\\prompts\\participant_prompts.json",
                        interactive=True
                    )
                    save_btn = gr.Button("Save Coordinates", variant="primary")
                    
                    # Unified status text
                    status_display = gr.Textbox(
                        value="Status: Ready. Please load frames to begin.",
                        label="Status",
                        interactive=False,
                        lines=2
                    )
        
        # ====================================================================
        # Event Handlers
        # ====================================================================
        
        def update_prompt_path_for_selection(project_name, participant_id, output_path, auto_prompt_path):
            """Update auto-filled prompt path while preserving manual edits."""
            selected_prompt_path, next_auto_path = resolve_prompt_path(
                project_name, participant_id, "", output_path, auto_prompt_path
            )
            return selected_prompt_path, next_auto_path

        def update_participants(project_name, output_path, auto_prompt_path):
            """Update participant dropdown when project changes."""
            if project_name:
                participants = get_participant_ids(project_name)
                default_val = participants[0] if participants else None
                selected_prompt_path, next_auto_path = resolve_prompt_path(
                    project_name, default_val, "", output_path, auto_prompt_path
                )
                return gr.update(choices=participants, value=default_val), selected_prompt_path, next_auto_path
            return gr.update(choices=[], value=None), "", ""

        project_dropdown.change(
            update_participants,
            inputs=[project_dropdown, output_path_input, auto_prompt_path_state],
            outputs=[participant_dropdown, output_path_input, auto_prompt_path_state]
        )
        participant_dropdown.change(
            update_prompt_path_for_selection,
            inputs=[project_dropdown, participant_dropdown, output_path_input, auto_prompt_path_state],
            outputs=[output_path_input, auto_prompt_path_state]
        )

        # Auto-populate participant dropdown on initial load
        interface.load(
            update_participants,
            inputs=[project_dropdown, output_path_input, auto_prompt_path_state],
            outputs=[participant_dropdown, output_path_input, auto_prompt_path_state]
        )
        
        def on_load_click(project_name, participant_id, manual_path, output_path, auto_prompt_path, ignore_existing_prompts=False):
            """Load frames for selected participant or manual path."""
            selected_prompt_path, next_auto_path = resolve_prompt_path(
                project_name, participant_id, manual_path, output_path, auto_prompt_path
            )
            if manual_path and manual_path.strip():
                frames_path = Path(manual_path.strip().strip('"'))
            else:
                if not project_name or not participant_id:
                    return (
                        None,
                        gr.update(minimum=0, value=0, maximum=1, interactive=False),
                        "Status: Error - Select project/participant or enter manual path",
                        0, {}, [], "", selected_prompt_path, next_auto_path, None
                    )
                frames_path = get_frames_path(project_name, participant_id)

            if not frames_path or not frames_path.exists():
                return (
                    None,
                    gr.update(minimum=0, value=0, maximum=1, interactive=False),
                    "Status: Error - Frames path not found",
                    0, {}, [], "", selected_prompt_path, next_auto_path, None
                )

            frames = get_frames_list(frames_path)
            if not frames:
                return (
                    None,
                    gr.update(minimum=0, value=0, maximum=1, interactive=False),
                    "Status: Error - No frames found in directory",
                    0, {}, [], "", selected_prompt_path, next_auto_path, None
                )

            if ignore_existing_prompts:
                dots_by_frame = {}
                if selected_prompt_path:
                    source_status = "Ignoring selected prompt JSON; starting empty."
                    prompt_status = "Saving will still use the selected prompt path."
                else:
                    source_status = "No prompt path selected; starting empty."
                    prompt_status = ""
            else:
                source_status = "Using selected prompt path." if selected_prompt_path else "No prompt path selected; starting empty."
                dots_by_frame, prompt_status = load_prompt_coordinates(selected_prompt_path, frames)

            cached = get_cached_display_frame(frames_path, frames[0])
            if cached is None:
                print(f"Error: Could not load first frame {frames[0]}")
                return (
                    None,
                    gr.update(minimum=0, value=0, maximum=1, interactive=False),
                    "Status: Error - Could not load first frame",
                    0, {}, [], "", selected_prompt_path, next_auto_path, None
                )

            display_img, orig_w, orig_h, disp_w, disp_h, _ = cached
            display_img = draw_annotations(display_img, get_frame_dots(dots_by_frame, 0), orig_w, orig_h, disp_w, disp_h)
            schedule_prefetch(frames_path, frames, 0)

            if manual_path and manual_path.strip():
                print(f"Loaded manual path {frames_path}: found {len(frames)} frames")
            else:
                print(f"Loaded {project_name}/{participant_id}: found {len(frames)} frames")

            return (
                display_img,
                gr.update(minimum=0, value=0, maximum=max(1, len(frames) - 1), interactive=True),
                f"Status: Successfully loaded {len(frames)} frames. {source_status} {prompt_status}",
                0,
                dots_by_frame,
                frames,
                str(frames_path),
                selected_prompt_path,
                next_auto_path,
                None
            )

        load_btn.click(
            on_load_click,
            inputs=[project_dropdown, participant_dropdown, manual_frames_input, output_path_input, auto_prompt_path_state],
            outputs=[
                image_display,
                frame_slider,
                status_display,
                current_frame_state,
                dots_by_frame_state,
                frames_list_state,
                active_frames_path_state,
                output_path_input,
                auto_prompt_path_state,
                pending_box_start_state
            ]
        )
        
        load_empty_btn.click(
            lambda project_name, participant_id, manual_path, output_path, auto_prompt_path: on_load_click(
                project_name,
                participant_id,
                manual_path,
                output_path,
                auto_prompt_path,
                ignore_existing_prompts=True,
            ),
            inputs=[project_dropdown, participant_dropdown, manual_frames_input, output_path_input, auto_prompt_path_state],
            outputs=[
                image_display,
                frame_slider,
                status_display,
                current_frame_state,
                dots_by_frame_state,
                frames_list_state,
                active_frames_path_state,
                output_path_input,
                auto_prompt_path_state,
                pending_box_start_state
            ]
        )
        def on_image_click(evt: gr.SelectData, dots_by_frame, active_frames_path, frame_idx, frames_list, is_positive, is_box_mode, pending_box_start, active_label):
            """Handle clicks on the image to add point prompts or create/edit box prompts."""
            frames_list = coerce_frames_list(frames_list)
            if not active_frames_path or not frames_list:
                return dots_by_frame, None, gr.update(), "Status: No folder loaded.", pending_box_start

            try:
                frame_idx = int(frame_idx)
            except (TypeError, ValueError):
                frame_idx = 0
            frame_idx = max(0, min(len(frames_list) - 1, frame_idx))

            frames_path = Path(active_frames_path)
            frame_name = frames_list[frame_idx]
            cached = get_cached_display_frame(frames_path, frame_name)
            if cached is None:
                return dots_by_frame, None, gr.update(), f"Status: Error - Could not load frame {frame_name}", pending_box_start

            display_img, orig_w, orig_h, disp_w, disp_h, _ = cached
            click_x, click_y = evt.index
            orig_x = int(click_x * orig_w / disp_w)
            orig_y = int(click_y * orig_h / disp_h)
            orig_x = max(0, min(orig_w - 1, orig_x))
            orig_y = max(0, min(orig_h - 1, orig_y))

            dots_by_frame = dict(dots_by_frame or {})
            frame_dots = list(get_frame_dots(dots_by_frame, frame_idx))
            label = active_label if active_label else ""

            if is_box_mode:
                pending = pending_box_start if isinstance(pending_box_start, dict) else None
                pending_matches = (
                    pending
                    and pending.get("frame") == frame_idx
                    and pending.get("label", "") == label
                )
                _, existing_box = find_box_for_label(frame_dots, label)

                if pending_matches and pending.get("mode") == "edit":
                    try:
                        corner_idx = int(pending["corner"])
                    except (KeyError, TypeError, ValueError):
                        corner_idx = -1
                    new_box = move_box_corner(pending.get("box"), corner_idx, orig_x, orig_y)
                    if not valid_box(new_box):
                        annotated_frame = draw_annotations(display_img, frame_dots, orig_w, orig_h, disp_w, disp_h)
                        annotated_frame = draw_pending_box_overlay(annotated_frame, pending, orig_w, orig_h, disp_w, disp_h)
                        return (
                            dots_by_frame,
                            annotated_frame,
                            update_coord_count(dots_by_frame, frame_idx),
                            "Status: Edited box is too small. Click a different corner location.",
                            pending,
                        )

                    frame_dots = remove_box_for_label(frame_dots, label)
                    frame_dots.append({"type": "box", "box": new_box, "label": label})
                    dots_by_frame[frame_idx] = frame_dots
                    dots_by_frame.pop(str(frame_idx), None)
                    annotated_frame = draw_annotations(display_img, frame_dots, orig_w, orig_h, disp_w, disp_h)
                    status = (
                        f"Status: Resized box for {label or 'selected label'} on frame {frame_idx + 1} / {len(frames_list)} "
                        f"({frame_name}); box={new_box}"
                    )
                    return dots_by_frame, annotated_frame, update_coord_count(dots_by_frame, frame_idx), status, None

                if existing_box is not None:
                    scale_x = disp_w / orig_w if orig_w else 1.0
                    scale_y = disp_h / orig_h if orig_h else 1.0
                    corner_idx = nearest_box_corner(existing_box, click_x, click_y, scale_x, scale_y)
                    if corner_idx is not None:
                        pending = {
                            "mode": "edit",
                            "frame": frame_idx,
                            "label": label,
                            "corner": corner_idx,
                            "box": existing_box,
                        }
                        annotated_frame = draw_annotations(display_img, frame_dots, orig_w, orig_h, disp_w, disp_h)
                        annotated_frame = draw_pending_box_overlay(annotated_frame, pending, orig_w, orig_h, disp_w, disp_h)
                        return (
                            dots_by_frame,
                            annotated_frame,
                            update_coord_count(dots_by_frame, frame_idx),
                            f"Status: Selected box corner {corner_idx + 1} for {label or 'selected label'}. Click its new location.",
                            pending,
                        )

                if pending_matches and pending.get("mode") == "create":
                    box_points = list(pending.get("points") or [])
                else:
                    box_points = []

                box_points.append({"x": orig_x, "y": orig_y})
                pending = {"mode": "create", "frame": frame_idx, "label": label, "points": box_points}
                if len(box_points) < 4:
                    annotated_frame = draw_annotations(display_img, frame_dots, orig_w, orig_h, disp_w, disp_h)
                    annotated_frame = draw_pending_box_overlay(annotated_frame, pending, orig_w, orig_h, disp_w, disp_h)
                    status = (
                        f"Status: Box corner {len(box_points)} / 4 set for {label or 'selected label'} "
                        f"on frame {frame_idx + 1}. Click remaining corners."
                    )
                    return dots_by_frame, annotated_frame, update_coord_count(dots_by_frame, frame_idx), status, pending

                box = box_from_points(box_points)
                if not valid_box(box):
                    pending = {"mode": "create", "frame": frame_idx, "label": label, "points": box_points[:3]}
                    annotated_frame = draw_annotations(display_img, frame_dots, orig_w, orig_h, disp_w, disp_h)
                    annotated_frame = draw_pending_box_overlay(annotated_frame, pending, orig_w, orig_h, disp_w, disp_h)
                    return (
                        dots_by_frame,
                        annotated_frame,
                        update_coord_count(dots_by_frame, frame_idx),
                        "Status: Box is too small. Re-click the fourth corner farther away.",
                        pending,
                    )

                replaced = existing_box is not None
                frame_dots = remove_box_for_label(frame_dots, label)
                frame_dots.append({"type": "box", "box": box, "label": label})
                dots_by_frame[frame_idx] = frame_dots
                dots_by_frame.pop(str(frame_idx), None)

                annotated_frame = draw_annotations(display_img, frame_dots, orig_w, orig_h, disp_w, disp_h)
                action = "Replaced" if replaced else "Added"
                status = (
                    f"Status: {action} 4-corner box for {label or 'selected label'} on frame {frame_idx + 1} / {len(frames_list)} "
                    f"({frame_name}); box={box}"
                )
                return dots_by_frame, annotated_frame, update_coord_count(dots_by_frame, frame_idx), status, None

            positive = bool(is_positive)
            frame_dots.append({
                "type": "point",
                "x": orig_x,
                "y": orig_y,
                "positive": positive,
                "label": label
            })
            dots_by_frame[frame_idx] = frame_dots
            dots_by_frame.pop(str(frame_idx), None)

            annotated_frame = draw_annotations(display_img, frame_dots, orig_w, orig_h, disp_w, disp_h)
            mode = "positive" if positive else "negative"
            status = (
                f"Status: Added {mode} {label or 'point'} on frame {frame_idx + 1} / {len(frames_list)} "
                f"({frame_name}); display click ({int(click_x)}, {int(click_y)}) -> original ({orig_x}, {orig_y})"
            )
            return dots_by_frame, annotated_frame, update_coord_count(dots_by_frame, frame_idx), status, None
        image_display.select(
            on_image_click,
            inputs=[
                dots_by_frame_state,
                active_frames_path_state,
                frame_slider,
                frames_list_state,
                is_positive_state,
                is_box_mode_state,
                pending_box_start_state,
                active_label_radio,
            ],
            outputs=[dots_by_frame_state, image_display, coord_count, status_display, pending_box_start_state]
        )
        
        def on_frame_change(frame_idx, frames_list, active_frames_path, dots_by_frame):
            """Handle frame navigation."""
            annotated_frame, resolved_idx, status, count = render_frame_for_ui(
                frame_idx, frames_list, active_frames_path, dots_by_frame
            )
            if annotated_frame is None:
                return gr.update(), gr.update(), gr.update(), status
            return annotated_frame, resolved_idx, count, status

        def supported_event_kwargs(listener, **candidates):
            import inspect
            try:
                params = inspect.signature(listener).parameters
            except Exception:
                return {}
            return {key: value for key, value in candidates.items() if key in params}

        frame_slider.change(
            on_frame_change,
            inputs=[frame_slider, frames_list_state, active_frames_path_state, dots_by_frame_state],
            outputs=[image_display, current_frame_state, coord_count, status_display],
            **supported_event_kwargs(frame_slider.change, trigger_mode="always_last", show_progress="hidden")
        )

        # Navigation buttons
        def on_prev_click(frame_idx, frames_list, active_frames_path, dots_by_frame):
            target = max(0, int(frame_idx) - 1)
            annotated_frame, resolved_idx, status, count = render_frame_for_ui(
                target, frames_list, active_frames_path, dots_by_frame
            )
            if annotated_frame is None:
                return gr.update(value=target), gr.update(), gr.update(), gr.update(), status
            return gr.update(value=resolved_idx), annotated_frame, resolved_idx, count, status

        def on_next_click(frame_idx, frames_list, active_frames_path, dots_by_frame):
            frames_list = coerce_frames_list(frames_list)
            if not frames_list:
                return gr.update(value=0), gr.update(), gr.update(), gr.update(), "Status: No folder loaded."
            target = min(len(frames_list) - 1, int(frame_idx) + 1)
            annotated_frame, resolved_idx, status, count = render_frame_for_ui(
                target, frames_list, active_frames_path, dots_by_frame
            )
            if annotated_frame is None:
                return gr.update(value=target), gr.update(), gr.update(), gr.update(), status
            return gr.update(value=resolved_idx), annotated_frame, resolved_idx, count, status

        prev_btn.click(
            on_prev_click,
            inputs=[frame_slider, frames_list_state, active_frames_path_state, dots_by_frame_state],
            outputs=[frame_slider, image_display, current_frame_state, coord_count, status_display]
        )

        next_btn.click(
            on_next_click,
            inputs=[frame_slider, frames_list_state, active_frames_path_state, dots_by_frame_state],
            outputs=[frame_slider, image_display, current_frame_state, coord_count, status_display]
        )

        # Undo button
        def on_undo_click(dots_by_frame, active_frames_path, frame_idx, frames_list):
            """Undo the last dot placed on the current frame."""
            frames_list = coerce_frames_list(frames_list)
            if not active_frames_path or not frames_list:
                return dots_by_frame, gr.update()

            try:
                frame_idx = int(frame_idx)
            except (TypeError, ValueError):
                frame_idx = 0
            frame_idx = max(0, min(len(frames_list) - 1, frame_idx))

            frame_dots = list(get_frame_dots(dots_by_frame, frame_idx))
            if not frame_dots:
                return dots_by_frame, gr.update()

            dots_by_frame = dict(dots_by_frame or {})
            frame_dots.pop()
            dots_by_frame[frame_idx] = frame_dots
            dots_by_frame.pop(str(frame_idx), None)

            annotated_frame, _, _, _ = render_frame_for_ui(frame_idx, frames_list, active_frames_path, dots_by_frame)
            return dots_by_frame, annotated_frame if annotated_frame is not None else gr.update()

        undo_btn.click(
            on_undo_click,
            inputs=[dots_by_frame_state, active_frames_path_state, frame_slider, frames_list_state],
            outputs=[dots_by_frame_state, image_display]
        )

        # Mode buttons
        positive_btn.click(
            lambda: (True, False, None),
            inputs=[],
            outputs=[is_positive_state, is_box_mode_state, pending_box_start_state]
        )
        negative_btn.click(
            lambda: (False, False, None),
            inputs=[],
            outputs=[is_positive_state, is_box_mode_state, pending_box_start_state]
        )
        box_btn.click(
            lambda: (True, True, None),
            inputs=[],
            outputs=[is_positive_state, is_box_mode_state, pending_box_start_state]
        )
        
        # Labels update
        def update_labels_handler(labels_text):
            labels = [l.strip() for l in labels_text.split(',') if l.strip()]
            if not labels:
                return gr.update()
            return gr.update(choices=labels, value=labels[0])
            
        update_labels_btn.click(
            update_labels_handler,
            inputs=[labels_input],
            outputs=[active_label_radio]
        )
        
        def save_coords_handler(dots_by_frame, output_path, project_name, participant_id):
            """Handle save coordinates button."""
            if (not project_name or not participant_id) and (not output_path or not output_path.strip()):
                return "Status: Error - Select project/participant, or specify an output path."
            
            return save_coordinates(dots_by_frame, output_path, project_name, participant_id)
        
        save_btn.click(
            save_coords_handler,
            inputs=[dots_by_frame_state, output_path_input, project_dropdown, participant_dropdown],
            outputs=status_display
        )
        
        dots_by_frame_state.change(
            update_coord_count,
            inputs=[dots_by_frame_state, current_frame_state],
            outputs=coord_count
        )

    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False, server_name="127.0.0.1", js=KEYBOARD_NAV_JS)
