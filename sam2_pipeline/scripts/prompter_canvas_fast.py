
#!/usr/bin/env python3
"""
prompter_canvas_fast.py - Canvas/SVG ROI annotation tool for SAM2 prompts.

Keeps the project/participant workflow from prompter_gradio_fast.py, but replaces
Gradio Image.select box editing with a browser-native SVG overlay. Boxes are
created by drag-release, then moved or resized with handles. Saved JSON remains
compatible with segment_video_faster.py.

Usage:
    python prompter_canvas_fast.py
    python prompter_canvas_fast.py --port 7862
    python prompter_canvas_fast.py --host 0.0.0.0 --port 7870

If the preferred port is busy, the app automatically tries the next ports.
"""

import argparse
import base64
import html
import json
import os
import socket
import threading
import time
import warnings
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

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

import gradio as gr
from PIL import Image

from prompter_gradio_fast import (
    CONFIG,
    DEFAULT_LABELS,
    canonical_prompt_path,
    coerce_frames_list,
    get_cached_display_frame,
    get_frame_dots,
    get_frames_list,
    get_frames_path,
    get_participant_ids,
    get_project_names,
    load_prompt_coordinates,
    normalise_dots_by_frame,
    resolve_prompt_path,
    save_coordinates,
    schedule_prefetch,
)

DATA_URI_CACHE_SIZE = int(CONFIG.get("data_uri_cache_size", 96))
_DATA_URI_CACHE = OrderedDict()
_DATA_URI_LOCK = threading.RLock()

FRAME_PAYLOAD_ELEM_ID = "frame-payload-bridge"
PROMPTS_ELEM_ID = "prompts-bridge"
ACTIVE_LABEL_ELEM_ID = "active-label-bridge"
STATUS_ELEM_ID = "canvas-status-display"
COUNT_ELEM_ID = "canvas-prompt-count"


def image_to_data_uri(image: Image.Image) -> str:
    img = image.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_data_uri(frames_path: Path, frame_name: str, display_img: Image.Image) -> str:
    key = (str(frames_path), frame_name)
    with _DATA_URI_LOCK:
        if key in _DATA_URI_CACHE:
            _DATA_URI_CACHE.move_to_end(key)
            return _DATA_URI_CACHE[key]
    data_uri = image_to_data_uri(display_img)
    with _DATA_URI_LOCK:
        _DATA_URI_CACHE[key] = data_uri
        _DATA_URI_CACHE.move_to_end(key)
        while len(_DATA_URI_CACHE) > DATA_URI_CACHE_SIZE:
            _DATA_URI_CACHE.popitem(last=False)
    return data_uri


def prompts_to_json(dots_by_frame: Dict) -> str:
    normalised = normalise_dots_by_frame(dots_by_frame)
    serialisable = {str(frame_idx): coords for frame_idx, coords in sorted(normalised.items())}
    return json.dumps(serialisable, separators=(",", ":"))


def prompts_from_json(prompt_json: str) -> Dict[int, List[Dict]]:
    if not prompt_json or not str(prompt_json).strip():
        return {}
    try:
        data = json.loads(prompt_json)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return normalise_dots_by_frame(data)


def count_prompt_types(coords: List[Dict]) -> Tuple[int, int]:
    points = 0
    boxes = 0
    for coord in coords or []:
        if isinstance(coord, dict) and (coord.get("type") == "box" or "box" in coord or "bbox" in coord):
            boxes += 1
        else:
            points += 1
    return points, boxes


def prompt_count_text(prompt_json: str, frame_idx: int) -> str:
    dots_by_frame = prompts_from_json(prompt_json)
    current = get_frame_dots(dots_by_frame, frame_idx)
    current_points, current_boxes = count_prompt_types(current)
    total_points = 0
    total_boxes = 0
    for coords in dots_by_frame.values():
        points, boxes = count_prompt_types(coords)
        total_points += points
        total_boxes += boxes
    current_total = current_points + current_boxes
    total = total_points + total_boxes
    return (
        f"Current Frame: {current_total} prompts "
        f"({current_points} points, {current_boxes} boxes) | "
        f"Total: {total} prompts ({total_points} points, {total_boxes} boxes)"
    )


def empty_frame_payload(message: str) -> str:
    return json.dumps({"ok": False, "message": message}, separators=(",", ":"))



def canvas_html(payload_json: str) -> str:
    """Render the current frame shell directly so the image appears even before JS syncs."""
    try:
        payload = json.loads(payload_json or "{}")
    except Exception:
        payload = {"ok": False, "message": "Could not parse frame payload."}

    payload_text = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    if not payload.get("ok"):
        message = html.escape(str(payload.get("message") or "Load frames to begin."))
        return f'''
<div id="sam2-canvas-app" class="sam2-canvas-app">
  <template id="sam2-frame-payload-dom">{payload_text}</template>
  <div id="sam2-stage-empty" class="sam2-stage-empty">{message}</div>
  <div id="sam2-stage" class="sam2-stage" tabindex="0" hidden>
    <img id="sam2-frame-image" class="sam2-frame-image" alt="Current frame" draggable="false" />
    <svg id="sam2-overlay" class="sam2-overlay" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>
</div>'''

    image = html.escape(str(payload.get("image") or ""), quote=True)
    disp_w = int(payload.get("disp_w") or 0)
    disp_h = int(payload.get("disp_h") or 0)
    return f'''
<div id="sam2-canvas-app" class="sam2-canvas-app">
  <template id="sam2-frame-payload-dom">{payload_text}</template>
  <div id="sam2-stage-empty" class="sam2-stage-empty" hidden>Load frames to begin.</div>
  <div id="sam2-stage" class="sam2-stage" tabindex="0">
    <img id="sam2-frame-image" class="sam2-frame-image" src="{image}" width="{disp_w}" height="{disp_h}" alt="Current frame" draggable="false" />
    <svg id="sam2-overlay" class="sam2-overlay" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>
</div>'''
def build_frame_payload(active_frames_path: str, frames_list, frame_idx) -> Tuple[str, int, str]:
    frames = coerce_frames_list(frames_list)
    if not active_frames_path or not frames:
        return empty_frame_payload("No folder loaded."), 0, "Status: No folder loaded."

    try:
        resolved_idx = int(frame_idx)
    except (TypeError, ValueError):
        resolved_idx = 0
    resolved_idx = max(0, min(len(frames) - 1, resolved_idx))

    frames_path = Path(active_frames_path)
    frame_name = frames[resolved_idx]
    start = time.perf_counter()
    cached = get_cached_display_frame(frames_path, frame_name)
    if cached is None:
        return empty_frame_payload(f"Could not load frame {frame_name}."), resolved_idx, f"Status: Error - Could not load frame {frame_name}"

    display_img, orig_w, orig_h, disp_w, disp_h, cache_hit = cached
    data_uri = get_data_uri(frames_path, frame_name, display_img)
    schedule_prefetch(frames_path, frames, resolved_idx)

    elapsed_ms = (time.perf_counter() - start) * 1000
    source = "cache" if cache_hit else "disk"
    payload = {
        "ok": True,
        "image": data_uri,
        "frame_idx": resolved_idx,
        "frame_name": frame_name,
        "frame_count": len(frames),
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "disp_w": int(disp_w),
        "disp_h": int(disp_h),
    }
    status = f"Status: Frame {resolved_idx + 1} / {len(frames)}: {frame_name} ({source}, {elapsed_ms:.0f} ms)"
    return json.dumps(payload, separators=(",", ":")), resolved_idx, status


def supported_event_kwargs(listener, **candidates):
    import inspect
    try:
        params = inspect.signature(listener).parameters
    except Exception:
        return {}
    return {key: value for key, value in candidates.items() if key in params}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def port_is_available(host: str, port: int) -> bool:
    bind_host = host or "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, int(port)))
        except OSError:
            return False
    return True


def find_available_port(host: str, preferred_port: int, scan_count: int) -> int:
    scan_count = max(1, int(scan_count))
    for port in range(preferred_port, preferred_port + scan_count):
        if port_is_available(host, port):
            return port
    raise OSError(
        f"Cannot find empty port in range: {preferred_port}-{preferred_port + scan_count - 1}. "
        "Use --port PORT or set GRADIO_SERVER_PORT to choose a different range."
    )


def parse_launch_args(argv=None):
    parser = argparse.ArgumentParser(description="Launch the SAM2 canvas prompt annotator.")
    parser.add_argument(
        "--host",
        "--server-name",
        default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        help="Host/interface for Gradio to bind. Default: %(default)s",
    )
    parser.add_argument(
        "--port",
        "--server-port",
        type=int,
        default=env_int("GRADIO_SERVER_PORT", 7861),
        help="Preferred Gradio port. Can also be set with GRADIO_SERVER_PORT. Default: %(default)s",
    )
    parser.add_argument(
        "--port-scan",
        type=int,
        default=env_int("SAM2_PROMPTER_PORT_SCAN", 50),
        help="How many ports to try starting at --port unless --strict-port is set. Default: %(default)s",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Use exactly --port and let Gradio fail if it is occupied.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args(argv)


def launch_interface(argv=None):
    args = parse_launch_args(argv)
    server_port = args.port if args.strict_port else find_available_port(args.host, args.port, args.port_scan)
    if server_port != args.port:
        print(f"Port {args.port} is busy; using {server_port} instead.")
    interface = create_interface()
    interface.launch(
        share=args.share,
        server_name=args.host,
        server_port=server_port,
        js=CANVAS_JS,
        css=CANVAS_CSS,
    )

CANVAS_HTML = r"""
<div id="sam2-canvas-app" class="sam2-canvas-app">
  <div id="sam2-stage-empty" class="sam2-stage-empty">Load frames to begin.</div>
  <div id="sam2-stage" class="sam2-stage" tabindex="0" hidden>
    <img id="sam2-frame-image" class="sam2-frame-image" alt="Current frame" draggable="false" />
    <svg id="sam2-overlay" class="sam2-overlay" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>
</div>
"""

CANVAS_CSS = r"""
.bridge-hidden { display: none !important; }
#sam2-canvas-app { width: 100%; min-height: 260px; }
.sam2-stage-empty {
  min-height: 420px; display: flex; align-items: center; justify-content: center;
  color: #6b7280; background: #111827; border: 1px solid #374151;
  border-radius: 6px; font-size: 14px;
}
.sam2-stage-empty[hidden], .sam2-stage[hidden] { display: none !important; }
.sam2-stage {
  position: relative; width: fit-content; max-width: 100%; line-height: 0;
  background: #111827; border: 1px solid #374151; border-radius: 6px;
  overflow: hidden; user-select: none; touch-action: none;
}
.sam2-frame-image {
  display: block; max-width: 100%; height: auto; user-select: none;
  -webkit-user-drag: none; pointer-events: none;
}
.sam2-overlay {
  position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair;
  touch-action: none; user-select: none; pointer-events: all;
}
#positive-mode-btn.sam2-active button, #positive-mode-btn.sam2-active,
#negative-mode-btn.sam2-active button, #negative-mode-btn.sam2-active,
#box-mode-btn.sam2-active button, #box-mode-btn.sam2-active {
  outline: 2px solid #0ea5e9 !important; outline-offset: 1px;
}
"""

CANVAS_JS = r"""
() => {
  const SCRIPT_VERSION = "canvas-interactions-2026-06-24-2";
  if (window.__sam2CanvasTimer) clearInterval(window.__sam2CanvasTimer);
  window.__sam2CanvasInstalled = SCRIPT_VERSION;

  const state = { payload: null, prompts: {}, mode: "positive", drag: null,
    selectedLabel: "", lastPayloadValue: null, lastPromptValue: null };

  function byId(id) { return document.getElementById(id); }
  function canvasRoots() {
    const surface = byId("canvas-surface");
    const roots = surface ? Array.from(surface.querySelectorAll(".sam2-canvas-app")) : [];
    return roots.length ? roots : Array.from(document.querySelectorAll(".sam2-canvas-app"));
  }
  function canvasRoot() {
    const roots = canvasRoots();
    roots.forEach((root, index) => { root.style.display = index === roots.length - 1 ? "" : "none"; });
    return roots.length ? roots[roots.length - 1] : null;
  }
  function canvasById(id) {
    const root = canvasRoot();
    return root ? root.querySelector(`#${id}`) : null;
  }
  function componentInput(id) {
    const root = byId(id);
    return root ? root.querySelector("textarea, input") : null;
  }
  function componentButton(id) {
    const root = byId(id);
    return root ? (root.querySelector("button") || root) : null;
  }
  function nativeSet(element, value) {
    if (!element) return;
    const descriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(element), "value");
    if (descriptor && descriptor.set) descriptor.set.call(element, value);
    else element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }
  function getStage() { return canvasById("sam2-stage"); }
  function getEmpty() { return canvasById("sam2-stage-empty"); }
  function getImage() { return canvasById("sam2-frame-image"); }
  function getSvg() { return canvasById("sam2-overlay"); }
  function frameKey() { return state.payload && state.payload.ok ? String(state.payload.frame_idx) : "0"; }
  function currentPrompts() {
    const key = frameKey();
    if (!Array.isArray(state.prompts[key])) state.prompts[key] = [];
    return state.prompts[key];
  }
  function activeLabel() {
    const bridge = componentInput("active-label-bridge");
    if (bridge && bridge.value) state.selectedLabel = bridge.value;
    return state.selectedLabel || "";
  }
  function writePrompts() {
    const text = JSON.stringify(state.prompts || {});
    state.lastPromptValue = text;
    nativeSet(componentInput("prompts-bridge"), text);
    updateCount();
  }
  function setStatus(text) { nativeSet(componentInput("canvas-status-display"), text || ""); }
  function setCount(text) { nativeSet(componentInput("canvas-prompt-count"), text || ""); }
  function isBox(prompt) { return prompt && (prompt.type === "box" || Array.isArray(prompt.box) || Array.isArray(prompt.bbox)); }
  function normalBox(box) {
    if (!Array.isArray(box) || box.length !== 4) return null;
    const values = box.map((value) => Math.round(Number(value)));
    if (values.some((value) => !Number.isFinite(value))) return null;
    const [x1, y1, x2, y2] = values;
    return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
  }
  function validBox(box) {
    const b = normalBox(box);
    return !!(b && Math.abs(b[2] - b[0]) >= 2 && Math.abs(b[3] - b[1]) >= 2);
  }
  function getBox(prompt) { return normalBox(prompt && (prompt.box || prompt.bbox)); }
  function boxHandles(box) {
    const [x1, y1, x2, y2] = box;
    return [[x1,y1,"nw"],[(x1+x2)/2,y1,"n"],[x2,y1,"ne"],[x2,(y1+y2)/2,"e"],
      [x2,y2,"se"],[(x1+x2)/2,y2,"s"],[x1,y2,"sw"],[x1,(y1+y2)/2,"w"]];
  }
  function svgScale() {
    const svg = getSvg();
    if (!svg || !state.payload || !state.payload.ok) return 1;
    const rect = svg.getBoundingClientRect();
    return rect.width && state.payload.orig_w ? state.payload.orig_w / rect.width : 1;
  }
  function overlayPoint(event) {
    const svg = getSvg();
    if (!svg || !state.payload || !state.payload.ok) return null;
    const rect = svg.getBoundingClientRect();
    const x = (event.clientX - rect.left) * state.payload.orig_w / rect.width;
    const y = (event.clientY - rect.top) * state.payload.orig_h / rect.height;
    return { x: Math.max(0, Math.min(state.payload.orig_w - 1, Math.round(x))),
      y: Math.max(0, Math.min(state.payload.orig_h - 1, Math.round(y))) };
  }
  function findBoxIndex(label) {
    const prompts = currentPrompts();
    for (let i = prompts.length - 1; i >= 0; i -= 1) {
      if (isBox(prompts[i]) && (prompts[i].label || "") === label) return i;
    }
    return -1;
  }
  function setBox(label, box) {
    const prompts = currentPrompts();
    const idx = findBoxIndex(label);
    const prompt = { type: "box", box: normalBox(box), label };
    if (idx >= 0) prompts[idx] = prompt; else prompts.push(prompt);
  }
  function makeSvg(name, attrs = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  }
  function updateCount() {
    if (!state.payload || !state.payload.ok) { setCount("0 prompts"); return; }
    let cp = 0, cb = 0, tp = 0, tb = 0;
    for (const [key, prompts] of Object.entries(state.prompts || {})) {
      for (const prompt of prompts || []) {
        if (isBox(prompt)) { tb += 1; if (key === frameKey()) cb += 1; }
        else { tp += 1; if (key === frameKey()) cp += 1; }
      }
    }
    setCount(`Current Frame: ${cp + cb} prompts (${cp} points, ${cb} boxes) | Total: ${tp + tb} prompts (${tp} points, ${tb} boxes)`);
  }
  function render() {
    const stage = getStage(), empty = getEmpty(), img = getImage(), svg = getSvg();
    if (!stage || !empty || !img || !svg) return;
    if (!state.payload || !state.payload.ok) {
      stage.hidden = true; stage.style.display = "none";
      empty.hidden = false; empty.style.display = "flex";
      empty.textContent = state.payload && state.payload.message ? state.payload.message : "Load frames to begin.";
      updateCount(); return;
    }
    empty.hidden = true; empty.style.display = "none";
    stage.hidden = false; stage.style.display = "";
    img.src = state.payload.image;
    img.width = state.payload.disp_w; img.height = state.payload.disp_h;
    svg.setAttribute("viewBox", `0 0 ${state.payload.orig_w} ${state.payload.orig_h}`);
    svg.innerHTML = "";
    svg.appendChild(makeSvg("rect", {
      x:0, y:0, width:state.payload.orig_w, height:state.payload.orig_h,
      fill:"transparent", "pointer-events":"all", "data-kind":"backdrop"
    }));
    const label = activeLabel();
    const scale = svgScale();
    const pointR = Math.max(3, 5 * scale);
    const handleR = Math.max(6, 8 * scale);

    for (const prompt of currentPrompts()) {
      if (isBox(prompt)) {
        const box = getBox(prompt); if (!box) continue;
        const [x1, y1, x2, y2] = box;
        const isActive = (prompt.label || "") === label;
        svg.appendChild(makeSvg("rect", { x:x1, y:y1, width:x2-x1, height:y2-y1,
          fill:isActive ? "rgba(6,182,212,0.10)" : "rgba(148,163,184,0.08)",
          stroke:isActive ? "#06b6d4" : "#94a3b8", "stroke-width":Math.max(2*scale,2),
          "data-kind":"box", "data-label":prompt.label || "", "vector-effect":"non-scaling-stroke" }));
        const text = makeSvg("text", { x:x1+4*scale, y:Math.max(14*scale, y1-4*scale),
          fill:isActive ? "#06b6d4" : "#94a3b8", "font-size":13*scale, "pointer-events":"none" });
        text.textContent = `${prompt.label || "label"} box`; svg.appendChild(text);
        if (isActive) {
          for (const [cx, cy, handle] of boxHandles(box)) {
            svg.appendChild(makeSvg("rect", { x:cx-handleR, y:cy-handleR, width:handleR*2, height:handleR*2,
              rx:1.5*scale, fill:"#06b6d4", stroke:"#111827", "stroke-width":Math.max(2*scale,2),
              "data-kind":"handle", "data-handle":handle, "data-label":prompt.label || "", "vector-effect":"non-scaling-stroke" }));
          }
        }
        continue;
      }
      const positive = prompt.positive === undefined || prompt.positive === true || prompt.positive === 1 || prompt.positive === "1";
      const color = positive ? "#22c55e" : "#ef4444";
      svg.appendChild(makeSvg("circle", { cx:Number(prompt.x)||0, cy:Number(prompt.y)||0, r:pointR,
        fill:color, stroke:"white", "stroke-width":Math.max(2*scale,2), "vector-effect":"non-scaling-stroke" }));
      if (prompt.label) {
        const t = makeSvg("text", { x:(Number(prompt.x)||0)+pointR+4*scale, y:(Number(prompt.y)||0)-pointR,
          fill:color, "font-size":12*scale, "pointer-events":"none" });
        t.textContent = prompt.label; svg.appendChild(t);
      }
    }
    if (state.drag && state.drag.liveBox) {
      const [x1, y1, x2, y2] = state.drag.liveBox;
      svg.appendChild(makeSvg("rect", { x:x1, y:y1, width:x2-x1, height:y2-y1,
        fill:"rgba(250,204,21,0.10)", stroke:"#facc15", "stroke-width":Math.max(2*scale,2),
        "stroke-dasharray":`${6*scale} ${4*scale}`, "vector-effect":"non-scaling-stroke", "pointer-events":"none" }));
    }
    updateCount();
  }

  function boxFromDrag(start, current) { return normalBox([start.x, start.y, current.x, current.y]); }
  function moveBox(box, dx, dy) {
    const [x1, y1, x2, y2] = box; const w = x2 - x1, h = y2 - y1;
    let nx1 = Math.max(0, Math.min(state.payload.orig_w - w, x1 + dx));
    let ny1 = Math.max(0, Math.min(state.payload.orig_h - h, y1 + dy));
    return normalBox([nx1, ny1, nx1 + w, ny1 + h]);
  }
  function resizeBox(box, handle, point) {
    let [x1, y1, x2, y2] = box;
    if (handle.includes("w")) x1 = point.x; if (handle.includes("e")) x2 = point.x;
    if (handle.includes("n")) y1 = point.y; if (handle.includes("s")) y2 = point.y;
    return normalBox([x1, y1, x2, y2]);
  }
  function removeActiveBox() {
    const label = activeLabel(); const prompts = currentPrompts(); const idx = findBoxIndex(label);
    if (idx < 0) { setStatus(`Status: No box for ${label || "selected label"} on this frame.`); return; }
    prompts.splice(idx, 1); writePrompts(); render();
    setStatus(`Status: Deleted box for ${label || "selected label"} on frame ${state.payload.frame_idx + 1}.`);
  }
  function undoLastPrompt() {
    const prompts = currentPrompts();
    if (!prompts.length) { setStatus("Status: Nothing to undo on this frame."); return; }
    const removed = prompts.pop(); writePrompts(); render();
    setStatus(`Status: Removed last ${isBox(removed) ? "box" : "point"} prompt on frame ${state.payload.frame_idx + 1}.`);
  }
  function onPointerDown(event) {
    if (!state.payload || !state.payload.ok || (event.button !== undefined && event.button !== 0)) return;
    const point = overlayPoint(event); if (!point) return;
    const svg = getSvg(); event.preventDefault(); event.stopPropagation();
    if (svg.setPointerCapture) svg.setPointerCapture(event.pointerId);
    const label = activeLabel(); const target = event.target;
    const kind = target && target.dataset ? target.dataset.kind : "";
    const handle = target && target.dataset ? target.dataset.handle : "";
    const targetLabel = target && target.dataset ? target.dataset.label || "" : "";
    if (state.mode === "positive" || state.mode === "negative") {
      currentPrompts().push({ type:"point", x:point.x, y:point.y, positive:state.mode === "positive", label });
      writePrompts(); render();
      setStatus(`Status: Added ${state.mode} ${label || "point"} on frame ${state.payload.frame_idx + 1} (${state.payload.frame_name}); original (${point.x}, ${point.y})`);
      return;
    }
    const prompts = currentPrompts(); const idx = findBoxIndex(label); const existingBox = idx >= 0 ? getBox(prompts[idx]) : null;
    if (kind === "handle" && targetLabel === label && existingBox) {
      state.drag = { type:"resize", label, handle, start:point, box:existingBox, liveBox:existingBox }; render(); return;
    }
    if (kind === "box" && targetLabel === label && existingBox) {
      state.drag = { type:"move", label, start:point, box:existingBox, liveBox:existingBox }; render(); return;
    }
    state.drag = { type:"create", label, start:point, liveBox:[point.x, point.y, point.x, point.y], replaced:existingBox !== null };
    render();
  }
  function onPointerMove(event) {
    if (!state.drag || !state.payload || !state.payload.ok) return;
    const point = overlayPoint(event); if (!point) return;
    event.preventDefault(); event.stopPropagation();
    if (state.drag.type === "create") state.drag.liveBox = boxFromDrag(state.drag.start, point);
    else if (state.drag.type === "move") state.drag.liveBox = moveBox(state.drag.box, point.x - state.drag.start.x, point.y - state.drag.start.y);
    else if (state.drag.type === "resize") state.drag.liveBox = resizeBox(state.drag.box, state.drag.handle, point);
    render();
  }
  function onPointerUp(event) {
    if (!state.drag || !state.payload || !state.payload.ok) return;
    event.preventDefault(); event.stopPropagation();
    const drag = state.drag; state.drag = null; const box = normalBox(drag.liveBox);
    if (!validBox(box)) { render(); setStatus("Status: Box was too small; drag a larger rectangle."); return; }
    setBox(drag.label, box); writePrompts(); render();
    let action = drag.type === "move" ? "Moved" : (drag.type === "resize" ? "Resized" : (drag.replaced ? "Replaced" : "Added"));
    setStatus(`Status: ${action} box for ${drag.label || "selected label"} on frame ${state.payload.frame_idx + 1} (${state.payload.frame_name}); box=[${box.join(", ")}]`);
  }
  function onPointerCancel(event) { if (!state.drag) return; event.preventDefault(); state.drag = null; render(); setStatus("Status: Box edit cancelled."); }
  function setMode(mode) {
    state.mode = mode; state.drag = null;
    for (const [id, value] of [["positive-mode-btn","positive"],["negative-mode-btn","negative"],["box-mode-btn","box"]]) {
      const root = byId(id); if (root) root.classList.toggle("sam2-active", value === mode);
    }
    setStatus(`Status: ${mode === "box" ? "Draw/Edit Box" : (mode === "positive" ? "Add Positive" : "Add Negative")} mode selected.`);
  }
  function stepFrame(delta) {
    const root = byId("frame-slider"); if (!root) return false;
    const input = root.querySelector('input[type="range"], input[type="number"]');
    if (!input || input.disabled) return false;
    const min = Number(input.min || 0), max = Number(input.max || 0), current = Number(input.value || 0);
    const next = Math.max(min, Math.min(max, current + delta)); if (next === current) return true;
    nativeSet(input, String(next)); return true;
  }
  function embeddedPayloadValue() {
    const domPayload = canvasById("sam2-frame-payload-dom");
    if (!domPayload) return "";
    if (domPayload.content && domPayload.content.textContent) return domPayload.content.textContent.trim();
    return (domPayload.innerHTML || domPayload.textContent || "").trim();
  }
  function parsePayload(value) {
    if (!value) return null;
    try { return JSON.parse(value); }
    catch (_) { return null; }
  }
  function syncFromBridges() {
    const payloadInput = componentInput("frame-payload-bridge");
    const bridgeValue = (payloadInput && payloadInput.value) || "";
    const domValue = embeddedPayloadValue();
    const domParsed = parsePayload(domValue);
    const bridgeParsed = parsePayload(bridgeValue);
    const payloadValue = domParsed && domParsed.ok ? domValue : (bridgeParsed && bridgeParsed.ok ? bridgeValue : (domValue || bridgeValue));
    const payloadParsed = payloadValue === domValue ? domParsed : (payloadValue === bridgeValue ? bridgeParsed : parsePayload(payloadValue));
    if (payloadValue && payloadValue !== state.lastPayloadValue) {
      state.lastPayloadValue = payloadValue;
      state.payload = payloadParsed || { ok:false, message:"Could not parse frame payload." };
      state.drag = null; render();
      if (state.payload && state.payload.ok) setStatus(`Status: Canvas ready on frame ${state.payload.frame_idx + 1}; ${state.mode === "box" ? "drag to draw/edit a box" : "click to add a point"}.`);
    }
    const promptsInput = componentInput("prompts-bridge");
    if (promptsInput && promptsInput.value !== state.lastPromptValue) {
      state.lastPromptValue = promptsInput.value;
      try {
        const parsed = JSON.parse(promptsInput.value || "{}");
        state.prompts = parsed && typeof parsed === "object" ? parsed : {};
      } catch (_) { state.prompts = {}; }
      state.drag = null; render();
    }
    activeLabel();
  }
  function install() {
    canvasRoot();
    const svg = getSvg();
    if (svg && svg.dataset.sam2Installed !== SCRIPT_VERSION) {
      svg.dataset.sam2Installed = SCRIPT_VERSION;
      svg.onpointerdown = onPointerDown;
      svg.onpointermove = onPointerMove;
      svg.onpointerup = onPointerUp;
      svg.onpointercancel = onPointerCancel;
      svg.ondragstart = (event) => event.preventDefault();
      svg.ondrop = (event) => event.preventDefault();
    }
    for (const [id, mode] of [["positive-mode-btn","positive"],["negative-mode-btn","negative"],["box-mode-btn","box"]]) {
      const btn = componentButton(id);
      if (btn && btn.dataset.sam2Installed !== "1") {
        btn.dataset.sam2Installed = "1";
        btn.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); setMode(mode); }, true);
      }
    }
    const deleteBtn = componentButton("delete-box-btn");
    if (deleteBtn && deleteBtn.dataset.sam2Installed !== "1") {
      deleteBtn.dataset.sam2Installed = "1";
      deleteBtn.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); removeActiveBox(); }, true);
    }
    const undoBtn = componentButton("undo-last-btn");
    if (undoBtn && undoBtn.dataset.sam2Installed !== "1") {
      undoBtn.dataset.sam2Installed = "1";
      undoBtn.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); undoLastPrompt(); }, true);
    }
  }
  document.addEventListener("keydown", (event) => {
    const tag = event.target && event.target.tagName;
    if (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tag)) return;
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      if (stepFrame(event.key === "ArrowRight" ? 1 : -1)) event.preventDefault(); return;
    }
    if (event.key === "Escape" && state.drag) { state.drag = null; render(); setStatus("Status: Box edit cancelled."); event.preventDefault(); return; }
    if ((event.key === "Delete" || event.key === "Backspace") && state.mode === "box") { removeActiveBox(); event.preventDefault(); }
  }, true);

  setMode("positive"); install(); syncFromBridges();
  window.__sam2CanvasTimer = setInterval(() => { install(); syncFromBridges(); }, 150);
}
"""

def create_interface():
    with gr.Blocks(title="SAM2 Canvas Prompter") as interface:
        gr.Markdown("# SAM2 Canvas Prompter: ROI Annotation")

        current_frame_state = gr.State(value=0)
        frames_list_state = gr.State(value=[])
        active_frames_path_state = gr.State(value="")
        auto_prompt_path_state = gr.State(value="")

        frame_payload_bridge = gr.Textbox(
            value=empty_frame_payload("Load frames to begin."),
            elem_id=FRAME_PAYLOAD_ELEM_ID,
            elem_classes=["bridge-hidden"],
            label="Frame Payload",
        )
        prompts_bridge = gr.Textbox(
            value="{}",
            elem_id=PROMPTS_ELEM_ID,
            elem_classes=["bridge-hidden"],
            label="Prompt Payload",
        )
        active_label_bridge = gr.Textbox(
            value=DEFAULT_LABELS[0] if DEFAULT_LABELS else "",
            elem_id=ACTIVE_LABEL_ELEM_ID,
            elem_classes=["bridge-hidden"],
            label="Active Label Bridge",
        )

        with gr.Row():
            with gr.Column(scale=2):
                canvas_display = gr.HTML(canvas_html(empty_frame_payload("Load frames to begin.")), elem_id="canvas-surface", js_on_load=f"({CANVAS_JS})();")

            with gr.Column(scale=1):
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
                        interactive=True,
                    )
                    participant_dropdown = gr.Dropdown(
                        choices=default_participants,
                        value=default_participant,
                        label="Participant ID",
                        interactive=True,
                    )
                    manual_frames_input = gr.Textbox(
                        label="Or Enter Manual Folder Path Directly",
                        placeholder="Y:\\...\\sam2_frames\\speech_task\\...",
                        interactive=True,
                    )
                    with gr.Row():
                        load_btn = gr.Button("Load Folder / Participant", variant="primary")
                        load_empty_btn = gr.Button("Load Empty (Ignore Prompts)")

                with gr.Group():
                    gr.Markdown("### 2. Navigate & Annotate")
                    with gr.Row():
                        labels_input = gr.Textbox(
                            label="Edit Labels (comma separated)",
                            value=",".join(DEFAULT_LABELS),
                            scale=3,
                        )
                        update_labels_btn = gr.Button("Update", scale=1)

                    active_label_radio = gr.Radio(
                        choices=DEFAULT_LABELS,
                        value=DEFAULT_LABELS[0] if DEFAULT_LABELS else "",
                        label="Select Active ROI Label",
                        elem_id="active-label-radio",
                    )

                    with gr.Row():
                        positive_btn = gr.Button("Add Positive", variant="primary", elem_id="positive-mode-btn")
                        negative_btn = gr.Button("Add Negative", variant="stop", elem_id="negative-mode-btn")
                    with gr.Row():
                        box_btn = gr.Button("Draw/Edit Box", elem_id="box-mode-btn")
                        delete_box_btn = gr.Button("Delete Box", elem_id="delete-box-btn")

                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=1,
                        label="Frame Number",
                        interactive=False,
                        elem_id="frame-slider",
                    )

                    with gr.Row():
                        prev_btn = gr.Button("Previous")
                        next_btn = gr.Button("Next")
                        undo_btn = gr.Button("Undo Last", elem_id="undo-last-btn")

                    coord_count = gr.Textbox(
                        value="0 prompts",
                        label="Prompt Count",
                        interactive=False,
                        elem_id=COUNT_ELEM_ID,
                    )

                with gr.Group():
                    gr.Markdown("### 3. Save Coordinates")
                    output_path_input = gr.Textbox(
                        value="",
                        label="Prompt JSON Path (editable; blank uses selection default)",
                        placeholder=canonical_prompt_path(default_project, default_participant) or "Y:\\...\\prompts\\participant_prompts.json",
                        interactive=True,
                    )
                    save_btn = gr.Button("Save Coordinates", variant="primary")
                    status_display = gr.Textbox(
                        value="Status: Ready. Please load frames to begin.",
                        label="Status",
                        interactive=False,
                        lines=2,
                        elem_id=STATUS_ELEM_ID,
                    )

        def update_prompt_path_for_selection(project_name, participant_id, output_path, auto_prompt_path):
            selected_prompt_path, next_auto_path = resolve_prompt_path(
                project_name, participant_id, "", output_path, auto_prompt_path
            )
            return selected_prompt_path, next_auto_path

        def update_participants(project_name, output_path, auto_prompt_path):
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
            outputs=[participant_dropdown, output_path_input, auto_prompt_path_state],
        )
        participant_dropdown.change(
            update_prompt_path_for_selection,
            inputs=[project_dropdown, participant_dropdown, output_path_input, auto_prompt_path_state],
            outputs=[output_path_input, auto_prompt_path_state],
        )
        interface.load(
            update_participants,
            inputs=[project_dropdown, output_path_input, auto_prompt_path_state],
            outputs=[participant_dropdown, output_path_input, auto_prompt_path_state],
        )

        def on_load_click(project_name, participant_id, manual_path, output_path, auto_prompt_path, ignore_existing_prompts=False):
            selected_prompt_path, next_auto_path = resolve_prompt_path(
                project_name, participant_id, manual_path, output_path, auto_prompt_path
            )
            if manual_path and manual_path.strip():
                frames_path = Path(manual_path.strip().strip('"'))
            else:
                if not project_name or not participant_id:
                    message = "Status: Error - Select project/participant or enter manual path"
                    return (
                        canvas_html(empty_frame_payload(message)), empty_frame_payload(message), "{}", gr.update(minimum=0, value=0, maximum=1, interactive=False),
                        message, 0, [], "", selected_prompt_path, next_auto_path, "0 prompts"
                    )
                frames_path = get_frames_path(project_name, participant_id)

            if not frames_path or not frames_path.exists():
                message = "Status: Error - Frames path not found"
                return (
                    canvas_html(empty_frame_payload(message)), empty_frame_payload(message), "{}", gr.update(minimum=0, value=0, maximum=1, interactive=False),
                    message, 0, [], "", selected_prompt_path, next_auto_path, "0 prompts"
                )

            frames = get_frames_list(frames_path)
            if not frames:
                message = "Status: Error - No frames found in directory"
                return (
                    canvas_html(empty_frame_payload(message)), empty_frame_payload(message), "{}", gr.update(minimum=0, value=0, maximum=1, interactive=False),
                    message, 0, [], "", selected_prompt_path, next_auto_path, "0 prompts"
                )

            if ignore_existing_prompts:
                dots_by_frame = {}
                prompt_status = "Ignoring selected prompt JSON; saving will still use that path." if selected_prompt_path else "Starting empty."
            else:
                dots_by_frame, prompt_status = load_prompt_coordinates(selected_prompt_path, frames)

            payload, resolved_idx, _ = build_frame_payload(str(frames_path), frames, 0)
            prompt_json = prompts_to_json(dots_by_frame)
            count = prompt_count_text(prompt_json, resolved_idx)

            if manual_path and manual_path.strip():
                print(f"Loaded manual path {frames_path}: found {len(frames)} frames")
            else:
                print(f"Loaded {project_name}/{participant_id}: found {len(frames)} frames")

            status = f"Status: Successfully loaded {len(frames)} frames. {prompt_status}"
            return (
                canvas_html(payload),
                payload,
                prompt_json,
                gr.update(minimum=0, value=0, maximum=max(1, len(frames) - 1), interactive=True),
                status,
                resolved_idx,
                frames,
                str(frames_path),
                selected_prompt_path,
                next_auto_path,
                count,
            )

        load_outputs = [
            canvas_display,
            frame_payload_bridge,
            prompts_bridge,
            frame_slider,
            status_display,
            current_frame_state,
            frames_list_state,
            active_frames_path_state,
            output_path_input,
            auto_prompt_path_state,
            coord_count,
        ]

        load_btn.click(
            on_load_click,
            inputs=[project_dropdown, participant_dropdown, manual_frames_input, output_path_input, auto_prompt_path_state],
            outputs=load_outputs,
        )
        load_empty_btn.click(
            lambda project_name, participant_id, manual_path, output_path, auto_prompt_path: on_load_click(
                project_name, participant_id, manual_path, output_path, auto_prompt_path, ignore_existing_prompts=True
            ),
            inputs=[project_dropdown, participant_dropdown, manual_frames_input, output_path_input, auto_prompt_path_state],
            outputs=load_outputs,
        )

        def on_frame_change(frame_idx, frames_list, active_frames_path, prompt_json):
            payload, resolved_idx, status = build_frame_payload(active_frames_path, frames_list, frame_idx)
            return canvas_html(payload), payload, resolved_idx, prompt_count_text(prompt_json, resolved_idx), status

        frame_slider.change(
            on_frame_change,
            inputs=[frame_slider, frames_list_state, active_frames_path_state, prompts_bridge],
            outputs=[canvas_display, frame_payload_bridge, current_frame_state, coord_count, status_display],
            **supported_event_kwargs(frame_slider.change, trigger_mode="always_last", show_progress="hidden"),
        )

        def on_prev_click(frame_idx, frames_list, active_frames_path, prompt_json):
            try:
                current = int(frame_idx)
            except (TypeError, ValueError):
                current = 0
            target = max(0, current - 1)
            payload, resolved_idx, status = build_frame_payload(active_frames_path, frames_list, target)
            return gr.update(value=resolved_idx), canvas_html(payload), payload, resolved_idx, prompt_count_text(prompt_json, resolved_idx), status

        def on_next_click(frame_idx, frames_list, active_frames_path, prompt_json):
            frames = coerce_frames_list(frames_list)
            if not frames:
                payload = empty_frame_payload("No folder loaded."); return gr.update(value=0), canvas_html(payload), payload, 0, "0 prompts", "Status: No folder loaded."
            try:
                current = int(frame_idx)
            except (TypeError, ValueError):
                current = 0
            target = min(len(frames) - 1, current + 1)
            payload, resolved_idx, status = build_frame_payload(active_frames_path, frames, target)
            return gr.update(value=resolved_idx), canvas_html(payload), payload, resolved_idx, prompt_count_text(prompt_json, resolved_idx), status

        prev_btn.click(
            on_prev_click,
            inputs=[frame_slider, frames_list_state, active_frames_path_state, prompts_bridge],
            outputs=[frame_slider, canvas_display, frame_payload_bridge, current_frame_state, coord_count, status_display],
        )
        next_btn.click(
            on_next_click,
            inputs=[frame_slider, frames_list_state, active_frames_path_state, prompts_bridge],
            outputs=[frame_slider, canvas_display, frame_payload_bridge, current_frame_state, coord_count, status_display],
        )

        def update_labels_handler(labels_text):
            labels = [label.strip() for label in labels_text.split(',') if label.strip()]
            if not labels:
                return gr.update(), gr.update()
            return gr.update(choices=labels, value=labels[0]), labels[0]

        update_labels_btn.click(
            update_labels_handler,
            inputs=[labels_input],
            outputs=[active_label_radio, active_label_bridge],
        )
        active_label_radio.change(
            lambda label: label or "",
            inputs=[active_label_radio],
            outputs=[active_label_bridge],
        )

        def save_coords_handler(prompt_json, output_path, project_name, participant_id):
            if (not project_name or not participant_id) and (not output_path or not output_path.strip()):
                return "Status: Error - Select project/participant, or specify an output path."
            dots_by_frame = prompts_from_json(prompt_json)
            return save_coordinates(dots_by_frame, output_path, project_name, participant_id)

        save_btn.click(
            save_coords_handler,
            inputs=[prompts_bridge, output_path_input, project_dropdown, participant_dropdown],
            outputs=status_display,
        )

    return interface


if __name__ == "__main__":
    launch_interface()

