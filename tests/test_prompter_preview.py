"""Unit tests for single-frame SAM 2 mask preview in prompter_gradio.py."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "sam2_pipeline", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import prompter_gradio as pg


def _make_dummy_frame(folder, name="00000.jpg", size=(160, 90)):
    img = Image.new("RGB", size, color=(40, 60, 80))
    img.save(os.path.join(folder, name))


class _FakePredictor:
    def __init__(self, device_type="cpu"):
        self.device = SimpleNamespace(type=device_type)
        self.image = None

    def set_image(self, img):
        self.image = img

    def predict(self, point_coords, point_labels, multimask_output=False):
        h, w = self.image.shape[:2]
        mask = np.zeros((1, h, w), dtype=bool)
        x = int(point_coords[0][0])
        y = int(point_coords[0][1])
        mask[0, max(0, y - 5):min(h, y + 6), max(0, x - 5):min(w, x + 6)] = True
        return mask, np.array([0.91], dtype=np.float32), None


def test_default_roi_colors_are_distinct():
    labels = ["Mother", "Child", "Judge_1", "Judge_2", "Door"]
    colors = [pg.get_roi_color(l, labels) for l in labels]
    assert len(set(colors)) == len(labels)


def test_preview_frame_masks_with_mocked_predictor():
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_frame(d)
        dots = {0: [(40, 30, 1, "Child"), (110, 50, 1, "Judge_2")]}
        with patch.object(pg, "get_image_predictor", return_value=_FakePredictor()):
            img, masks, status = pg.preview_frame_masks(
                d, ["00000.jpg"], 0, dots, {}, "facebook/sam2.1-hiera-large", "Mother,Child,Judge_1,Judge_2,Door"
            )
        assert img is not None
        assert 0 in masks and set(masks[0].keys()) == {"Child", "Judge_2"}
        assert "Previewed frame 1" in status
        assert "Child:" in status and "Judge_2:" in status


def test_preview_handles_missing_sam2_gracefully():
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_frame(d)
        dots = {0: [(40, 30, 1, "Mother")]}
        with patch.object(pg, "get_image_predictor", side_effect=ImportError("No module named 'sam2'")):
            img, masks, status = pg.preview_frame_masks(
                d, ["00000.jpg"], 0, dots, {}, "facebook/sam2.1-hiera-large", "Mother"
            )
        assert img is not None
        assert masks == {}
        assert "PyTorch and SAM 2" in status


def test_undo_last_clears_preview_when_frame_becomes_empty():
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_frame(d)
        dots = {0: [(40, 30, 1, "Mother")]}
        coords = [{"x": 40, "y": 30, "positive": 1, "label": "Mother", "frame": 0}]
        masks = {0: {"Mother": np.ones((90, 160), dtype=bool)}}
        img, new_dots, new_coords, new_masks, status = pg.undo_last(
            d, ["00000.jpg"], 0, dots, coords, masks, "Mother"
        )
        assert img is not None
        assert new_dots[0] == []
        assert new_coords == []
        assert 0 not in new_masks
        assert "Undid last point" in status


def test_save_and_reload_coordinates_schema():
    with tempfile.TemporaryDirectory() as d:
        _make_dummy_frame(d)
        out_json = os.path.join(d, "coordinates.json")
        coords = [{"x": 50, "y": 40, "positive": 1, "label": "Mother", "frame": 0}]
        msg = pg.save_coordinates(coords, "Mother,Child", out_json)
        assert "Saved 1 coordinates" in msg
        with open(out_json) as f:
            saved = json.load(f)
        assert "Mother" in saved and saved["Mother"][0]["x"] == 50

        _, _, _, _, status, dots, loaded_coords, _, _, _ = pg.load_folder(d, "Mother,Child", out_json)
        assert "coordinates.json" in status
        assert len(loaded_coords) == 1
        assert len(dots[0]) == 1
