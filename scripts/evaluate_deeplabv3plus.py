import csv
import json
import random
import re
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw
from remotezip import RemoteZip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader

BASE = "http://128.32.162.150/bdd100k/"
TRAIN_N, VAL_N = 400, 100
WIDTH, HEIGHT = 512, 288
SEED = 42
NUM_CLASSES = 19
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32)
]
IGNORE_COLOR = (0, 0, 0)


def stem(path):
    return PurePosixPath(path).stem


def mask_stem(path):
    return re.sub(r"_train_id$", "", stem(path), flags=re.I)


def prepare_validation():
    root = Path("evaluation-data/val")
    image_dir, mask_dir = root / "images", root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with RemoteZip(BASE + "bdd100k_images_10k.zip") as iz, RemoteZip(BASE + "bdd100k_seg_maps.zip") as mz:
        image_names, mask_names = iz.namelist(), mz.namelist()
        train_images = [p for p in image_names if p.lower().endswith((".jpg", ".jpeg")) and "/train/" in ("/" + p.lower())]
        val_images = [p for p in image_names if p.lower().endswith((".jpg", ".jpeg")) and "/val/" in ("/" + p.lower())]
        train_masks = [p for p in mask_names if p.lower().endswith("_train_id.png") and "/train/" in ("/" + p.lower())]
        val_masks = [p for p in mask_names if p.lower().endswith("_train_id.png") and "/val/" in ("/" + p.lower())]
        tmap = {mask_stem(p): p for p in train_masks}
        vmap = {mask_stem(p): p for p in val_masks}
        train_pairs = [(p, tmap[stem(p)]) for p in train_images if stem(p) in tmap]
        val_pairs = [(p, vmap[stem(p)]) for p in val_images if stem(p) in vmap]
        if len(train_pairs) != 7000 or len(val_pairs) != 1000:
            raise RuntimeError(f"BDD100K pairing mismatch: train={len(train_pairs)}, val={len(val_pairs)}")

        # Reproduce the exact deterministic sample selection from the training workflow.
        rng = random.Random(SEED)
        rng.shuffle(train_pairs)
        rng.shuffle(val_pairs)
        selected = val_pairs[:VAL_N]
        if len(selected) != VAL_N:
            raise RuntimeError(f"Expected {VAL_N} validation pairs, got {len(selected)}")

        ids_seen = set()
        for idx, (image_path, mask_path) in enumerate(selected, 1):
            out_image = image_dir / f"{stem(image_path)}.jpg"
            out_mask = mask_dir / f"{stem(image_path)}.png"
            if not out_image.exists():
                with Image.open(BytesIO(iz.read(image_path))) as image:
                    image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR).save(out_image, quality=95)
            if not out_mask.exists():
                with Image.open(BytesIO(mz.read(mask_path))) as mask:
                    if mask.mode not in ("L", "P", "I", "I;16"):
                        raise RuntimeError(f"Unexpected mask mode {mask.mode}: {mask_path}")
                    arr = np.asarray(mask)
                    ids = set(int(x) for x in np.unique(arr))
                    if not ids.issubset(set(range(NUM_CLASSES)) | {255}):
                        raise RuntimeError(f"Invalid mask IDs in {mask_path}: {sorted(ids)}")
                    ids_seen.update(ids)
                    mask.resize((WIDTH, HEIGHT), Image.Resampling.NEAREST).save(out_mask)
            if idx % 25 == 0:
                print(f"Prepared validation pairs {idx}/{VAL_N}", flush=True)

    report = {
        "status": "PASS",
        "train_pairs_verified": len(train_pairs),
        "validation_pairs_available": len(val_pairs),
        "validation_pairs_prepared": VAL_N,
        "seed": SEED,
        "resolution": [WIDTH, HEIGHT],
        "mask_ids_seen": sorted(ids_seen),
        "seconds": round(time.time() - started, 2)
    }
    Path("evaluation-results/data_validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            *[nn.Sequential(nn.Conv2d(2048, 256, 3, padding=d, dilation=d, bias=False),
                            nn.BatchNorm2d(256), nn.ReLU(inplace=True)) for d in (6, 12, 18)]
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2048, 256, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(1280, 256, 1, bias=False), nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.aspp = ASPP()
        self.low = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, NUM_CLASSES, 1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer4(self.layer3(self.layer2(low)))
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low(low)], dim=1))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class BDDValidation(Dataset):
    def __init__(self):
        self.images = sorted(Path("evaluation-data/val/images").glob("*.jpg"))
        self.mask_dir = Path("evaluation-data/val/masks")
        if len(self.images) != VAL_N:
            raise RuntimeError(f"Expected {VAL_N} images, found {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        mask = np.asarray(Image.open(self.mask_dir / (path.stem + ".png")), dtype=np.int64).copy()
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask), path.name


def colorize(mask):
    mask = np.asarray(mask)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        out[mask == class_id] = color
    out[mask == 255] = IGNORE_COLOR
    return Image.fromarray(out)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("num_classes") != NUM_CLASSES:
        raise RuntimeError(f"{path}: expected {NUM_CLASSES} classes, got {checkpoint.get('num_classes')}")
    if checkpoint.get("input_size") != [WIDTH, HEIGHT]:
        raise RuntimeError(f"{path}: unexpected input size {checkpoint.get('input_size')}")
    model = DeepLabV3Plus()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def evaluate_checkpoint(path, dataset, criterion, save_visuals=False):
    model, checkpoint = load_checkpoint(path)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    losses, visual_count = [], 0
    visual_dir = Path("evaluation-results/visual_examples")
    if save_visuals:
        visual_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for step, (images, targets, filenames) in enumerate(loader, 1):
            logits = model(images)
            losses.append(float(criterion(logits, targets)))
            predictions = logits.argmax(1)
            pred = predictions.reshape(-1)
            target = targets.reshape(-1)
            valid = target != 255
            confusion += torch.bincount(
                NUM_CLASSES * target[valid] + pred[valid],
                minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            if save_visuals and visual_count < 12 and (step - 1) % 8 == 0:
                original_path = Path("evaluation-data/val/images") / filenames[0]
                original = Image.open(original_path).convert("RGB")
                pred_color = colorize(predictions[0].cpu().numpy())
                gt_color = colorize(targets[0].cpu().numpy())
                panel = Image.new("RGB", (WIDTH * 3, HEIGHT + 34), (245, 245, 245))
                panel.paste(original, (0, 34))
                panel.paste(pred_color, (WIDTH, 34))
                panel.paste(gt_color, (WIDTH * 2, 34))
                draw = ImageDraw.Draw(panel)
                draw.text((8, 10), "Original", fill=(0, 0, 0))
                draw.text((WIDTH + 8, 10), "Predicción", fill=(0, 0, 0))
                draw.text((WIDTH * 2 + 8, 10), "Máscara real", fill=(0, 0, 0))
                panel.save(visual_dir / f"{visual_count + 1:02d}_{Path(filenames[0]).stem}.jpg", quality=90)
                visual_count += 1

            if step % 20 == 0 or step == len(loader):
                print(f"{path.name}: evaluación {step}/{len(loader)}", flush=True)

    intersection = confusion.diag().double()
    union = confusion.sum(0).double() + confusion.sum(1).double() - intersection
    target_count = confusion.sum(1).double()
    pred_count = confusion.sum(0).double()
    present = union > 0
    iou = intersection / union.clamp_min(1)
    precision = intersection / pred_count.clamp_min(1)
    recall = intersection / target_count.clamp_min(1)
    pixel_accuracy = intersection.sum() / confusion.sum().clamp_min(1)
    report = {
        "checkpoint": path.name,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_metrics_saved": checkpoint.get("metrics", {}),
        "validation_samples": len(dataset),
        "validation_loss": float(np.mean(losses)),
        "mean_iou_present_classes": float(iou[present].mean()),
        "mean_iou_all_19_classes": float(iou.mean()),
        "pixel_accuracy": float(pixel_accuracy),
        "valid_pixels": int(confusion.sum()),
        "classes": []
    }
    for idx, name in enumerate(CLASS_NAMES):
        report["classes"].append({
            "id": idx,
            "class": name,
            "iou": float(iou[idx]) if union[idx] > 0 else None,
            "precision": float(precision[idx]) if pred_count[idx] > 0 else None,
            "recall": float(recall[idx]) if target_count[idx] > 0 else None,
            "ground_truth_pixels": int(target_count[idx]),
            "predicted_pixels": int(pred_count[idx])
        })
    return report, confusion.tolist(), visual_count


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    Path("evaluation-results").mkdir(exist_ok=True)
    started = time.time()

    best_path = Path("training-results/deeplabv3plus_best.pth")
    last_path = Path("training-results/deeplabv3plus_last.pth")
    history_path = Path("training-results/training_history.json")
    training_report_path = Path("training-results/training_report.json")
    for required in (best_path, last_path, history_path, training_report_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing expected artifact file: {required}")

    training_report = json.loads(training_report_path.read_text())
    history = json.loads(history_path.read_text())
    print(f"Training artifact: {training_report.get('status')} | epochs={training_report.get('epochs_completed')} | best mIoU={training_report.get('best_miou')}", flush=True)

    prepare_validation()
    dataset = BDDValidation()
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_report, best_confusion, visual_count = evaluate_checkpoint(best_path, dataset, criterion, save_visuals=True)
    # Check the last checkpoint is structurally loadable too; no second full evaluation is needed.
    last_model, last_checkpoint = load_checkpoint(last_path)
    del last_model

    result = {
        "status": "PASS",
        "architecture": "DeepLabV3+ ResNet-50",
        "training_run_id": 45887532742,
        "training_artifact_id": 15157038640,
        "training_report": training_report,
        "history_epochs": len(history),
        "best_checkpoint_evaluation": best_report,
        "last_checkpoint_load_test": {
            "status": "PASS",
            "epoch": int(last_checkpoint["epoch"]),
            "optimizer_state_present": "optimizer_state_dict" in last_checkpoint
        },
        "visual_examples_saved": visual_count,
        "total_evaluation_seconds": round(time.time() - started, 2)
    }
    (Path("evaluation-results") / "evaluation_report.json").write_text(json.dumps(result, indent=2))

    with (Path("evaluation-results") / "class_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "iou", "precision", "recall", "ground_truth_pixels", "predicted_pixels"])
        writer.writeheader()
        writer.writerows(best_report["classes"])

    (Path("evaluation-results") / "confusion_matrix.json").write_text(json.dumps({
        "class_names": CLASS_NAMES,
        "matrix": best_confusion
    }, indent=2))

    print("EVALUATION FINAL REPORT", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
