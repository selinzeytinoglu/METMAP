"""Unit tests for SAM 2 predicted-IoU confidence output in segment_video.py."""

import json
import os
import tempfile

import torch


def build_sidecar(inference_state, obj_id_to_name, out_obj_ids, out_frame_idx):
    """Mirror of the segment_video.py per-frame predicted-IoU extraction."""
    iou_per_obj = inference_state.get("iou_pred_per_frame", {}).get(out_frame_idx, [])
    frame_conf = {}
    for i, out_obj_id in enumerate(out_obj_ids):
        iou = iou_per_obj[i] if i < len(iou_per_obj) else None
        frame_conf[obj_id_to_name[out_obj_id]] = (
            round(float(iou.reshape(-1).max().detach().cpu()), 4) if iou is not None else None
        )
    return {f"frame_{out_frame_idx:05d}": frame_conf}


def test_predicted_iou_sidecar():
    inference_state = {"iou_pred_per_frame": {}}
    inference_state["iou_pred_per_frame"][0] = [torch.tensor([[0.87]]), torch.tensor([[0.72]])]
    inference_state["iou_pred_per_frame"][1] = [torch.tensor([[0.55]]), None]

    obj_id_to_name = {0: "Parent", 1: "Child"}

    sidecar = {}
    sidecar.update(build_sidecar(inference_state, obj_id_to_name, [0, 1], 0))
    sidecar.update(build_sidecar(inference_state, obj_id_to_name, [0, 1], 1))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "predicted_iou.json")
        with open(path, "w") as f:
            json.dump(sidecar, f, indent=2)
        with open(path) as f:
            loaded = json.load(f)

    assert abs(float(loaded["frame_00000"]["Parent"]) - 0.87) < 1e-4
    assert abs(float(loaded["frame_00000"]["Child"]) - 0.72) < 1e-4
    assert abs(float(loaded["frame_00001"]["Parent"]) - 0.55) < 1e-4
    assert loaded["frame_00001"]["Child"] is None
    assert abs(float(torch.tensor([[0.9, 0.6]]).reshape(-1).max()) - 0.9) < 1e-6

    print("PASS: predicted-IoU JSON serialization")


def test_missing_frame_is_safe():
    inference_state = {"iou_pred_per_frame": {}}
    out = build_sidecar(inference_state, {0: "Parent"}, [0], 99)
    assert out["frame_00099"]["Parent"] is None
    print("PASS: missing-frame handling")


if __name__ == "__main__":
    test_predicted_iou_sidecar()
    test_missing_frame_is_safe()
    print("All predicted-IoU tests passed.")

