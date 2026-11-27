#!/usr/bin/env python3
"""Readiness tests for the remaining BDD100K project components.

This test deliberately uses synthetic tensors/labels: it validates the Mask R-CNN
implementation and BDD100K instance-bitmask decoder without downloading BDD100K
or consuming a long CPU training run.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CATEGORIES = {
    1: "pedestrian", 2: "rider", 3: "car", 4: "truck",
    5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle",
}


def decode_bdd_instance_rgba(rgba):
    """Decode BDD100K RGBA instance bitmask into per-instance binary masks.

    R stores category id, G stores flags (crowd bit 1, ignore bit 0),
    B stores instance id for the instance-segmentation task.
    """
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected RGBA mask, got shape={arr.shape}")
    category = arr[..., 0]
    flags = arr[..., 1]
    instance_id = arr[..., 2]
    valid = (category > 0) & ((flags & 1) == 0) & ((flags & 2) == 0)
    instances = []
    pairs = np.unique(np.stack([category[valid], instance_id[valid]], axis=-1), axis=0)
    for cat_id, inst_id in pairs:
        if int(cat_id) not in CATEGORIES or int(inst_id) == 0:
            continue
        mask = valid & (category == cat_id) & (instance_id == inst_id)
        if int(mask.sum()) == 0:
            continue
        ys, xs = np.where(mask)
        instances.append({
            "category_id": int(cat_id),
            "category": CATEGORIES[int(cat_id)],
            "instance_id": int(inst_id),
            "mask": mask,
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "area": int(mask.sum()),
        })
    return instances


def test_instance_decoder():
    rgba = np.zeros((32, 48, 4), dtype=np.uint8)
    rgba[3:15, 4:18, 0] = 3
    rgba[3:15, 4:18, 2] = 1
    rgba[18:29, 25:42, 0] = 1
    rgba[18:29, 25:42, 2] = 2
    rgba[0:2, 0:2, 0] = 3
    rgba[0:2, 0:2, 1] = 1  # ignore flag
    instances = decode_bdd_instance_rgba(rgba)
    assert len(instances) == 2, len(instances)
    assert {x["category"] for x in instances} == {"car", "pedestrian"}
    assert all(x["mask"].dtype == np.bool_ for x in instances)
    assert all(x["area"] > 0 for x in instances)
    return {"status": "PASS", "instances_decoded": len(instances),
            "categories": sorted(x["category"] for x in instances)}


def test_mask_rcnn():
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                  min_size=128, max_size=160,
                                  box_detections_per_img=10)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 9)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, 9)

    image = torch.rand(3, 128, 160)
    masks = torch.zeros((2, 128, 160), dtype=torch.uint8)
    masks[0, 20:60, 20:70] = 1
    masks[1, 65:110, 85:135] = 1
    target = {
        "boxes": torch.tensor([[20., 20., 70., 60.], [85., 65., 135., 110.]]),
        "labels": torch.tensor([3, 1], dtype=torch.int64),
        "masks": masks,
        "image_id": torch.tensor([1]),
        "area": torch.tensor([2000., 2250.]),
        "iscrowd": torch.zeros((2,), dtype=torch.int64),
    }
    model.train()
    losses = model([image], [target])
    total = sum(losses.values())
    assert torch.isfinite(total), {k: float(v.detach()) for k, v in losses.items()}
    total.backward()
    loss_values = {k: float(v.detach()) for k, v in losses.items()}

    model.eval()
    with torch.inference_mode():
        predictions = model([image])
    assert len(predictions) == 1
    assert set(("boxes", "labels", "scores", "masks")).issubset(predictions[0])
    assert predictions[0]["masks"].ndim == 4
    return {"status": "PASS", "num_classes_including_background": 9,
            "losses": loss_values, "prediction_fields": sorted(predictions[0].keys()),
            "predicted_instances": int(len(predictions[0]["boxes"]))}


def test_environment_join():
    # Validate strict filename matching and preserve unknown metadata.
    image_names = {"a.jpg", "b.jpg", "c.jpg", "unmatched.jpg"}
    labels = [
        {"name": "a.jpg", "attributes": {"weather": "rainy", "timeofday": "night", "scene": "city street"}},
        {"name": "b.jpg", "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "highway"}},
        {"name": "c.jpg", "attributes": {"weather": "foggy", "timeofday": "dawn/dusk", "scene": "residential"}},
    ]
    by_name = {Path(x["name"]).name: x.get("attributes", {}) for x in labels}
    rows = []
    for name in sorted(image_names):
        attrs = by_name.get(name, {})
        rows.append({"image": name, "weather": attrs.get("weather", "unknown"),
                     "timeofday": attrs.get("timeofday", "unknown"),
                     "scene": attrs.get("scene", "unknown")})
    matched = sum(x["weather"] != "unknown" for x in rows)
    assert matched == 3
    assert next(x for x in rows if x["image"] == "unmatched.jpg")["weather"] == "unknown"
    return {"status": "PASS", "images": len(rows), "matched": matched,
            "unmatched_preserved_as_unknown": len(rows) - matched,
            "example_rows": rows}


def main():
    report = {
        "scope": "synthetic readiness tests; not a real BDD100K training/evaluation",
        "instance_bitmask_decoder": test_instance_decoder(),
        "mask_rcnn_forward_backward_and_inference": test_mask_rcnn(),
        "environmental_metadata_join": test_environment_join(),
    }
    report["status"] = "PASS" if all(
        report[k]["status"] == "PASS" for k in (
            "instance_bitmask_decoder", "mask_rcnn_forward_backward_and_inference",
            "environmental_metadata_join")) else "FAIL"
    Path("component-readiness").mkdir(exist_ok=True)
    Path("component-readiness/report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
