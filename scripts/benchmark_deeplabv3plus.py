#!/usr/bin/env python3
"""CPU smoke test and throughput benchmark for the project's DeepLabV3+ model.

Uses synthetic tensors intentionally: BDD100K is not stored in GitHub.
This validates model execution and estimates compute throughput, not data-loader
or end-to-end dataset performance.
"""
import argparse
import json
import math
import os
import resource
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
            *[
                nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                         padding=d, dilation=d, bias=False),
                              nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
                for d in (6, 12, 18)
            ],
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                  nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )

    def forward(self, x):
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:],
                               mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(b(x) for b in self.branches), pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=19, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights,
                            replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                   backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = ASPP(2048, 256)
        self.low_projection = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.aspp(self.layer4(x))
        high = F.interpolate(high, size=low.shape[-2:],
                             mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([high, self.low_projection(low)], dim=1))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = DeepLabV3Plus(num_classes=19, pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    x = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    y = torch.randint(0, 19, (args.batch_size, args.height, args.width),
                      dtype=torch.long, device=device)

    # Warm-up is included in total wall time but excluded from step average.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    assert out.shape == (args.batch_size, 19, args.height, args.width), out.shape
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    durations, losses = [], []
    for _ in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        durations.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))

    step_seconds = sum(durations) / len(durations)
    # Planning scenario discussed with the user: 800 train images, batch 1.
    train_images = 800
    steps_per_epoch = math.ceil(train_images / args.batch_size)
    epochs = 30
    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "cpu_count": os.cpu_count(),
        "batch_size": args.batch_size,
        "input_size": [args.height, args.width],
        "benchmark_steps": args.steps,
        "seconds_per_step_mean": step_seconds,
        "seconds_per_step_each": durations,
        "last_loss": losses[-1],
        "output_shape": list(out.shape),
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "planning_train_images": train_images,
        "planning_epochs": epochs,
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_seconds_per_epoch": step_seconds * steps_per_epoch,
        "estimated_hours_30_epochs":
            step_seconds * steps_per_epoch * epochs / 3600,
        "note": "Synthetic tensors; excludes BDD100K I/O, validation, checkpoints and setup."
    }
    Path("benchmark-results").mkdir(exist_ok=True)
    Path("benchmark-results/cpu_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
