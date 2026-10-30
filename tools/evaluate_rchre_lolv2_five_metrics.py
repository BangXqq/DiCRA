#!/usr/bin/env python3
"""Add patched-Inception FID and normalized RGB MAE to locked 3-seed LOLv2 evaluation."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import linalg
import torch
from basicsr.archs.inception import InceptionV3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_lsrw_official_checkpoint_fusion import tensor
from tools.train_reliability_residual_gate import (
    ReliabilityResidualGate, apply_gate as apply_rsra, ALPHA)
from tools.train_rchre_validation import RCHRE

MODEL = ROOT / "experiments/rchre_validation_v1/model.pt"
RSRA = ROOT / "experiments/reliability_residual_gate_v1/gate.pt"
SOURCE = ROOT / "reproduction_records/rchre_lolv2_locked_20260922/summary.json"
OUT = ROOT / "reproduction_records/rchre_lolv2_five_metrics_20260923"
SETS = {
    "Real": {"low": ROOT / "dataset/LOLv2_Real/Test/Low",
             "gt": ROOT / "dataset/LOLv2_Real/Test/Normal",
             "prior": ROOT / "experiments/retinexformer_lolv2_real_official_prior",
             "result": "Diff-Retinex_Plus_lolv2_real_test", "prefix": "normal"},
    "Synthetic": {"low": ROOT / "dataset/LOLv2_Synthetic/Test/Low",
                  "gt": ROOT / "dataset/LOLv2_Synthetic/Test/Normal",
                  "prior": ROOT / "experiments/retinexformer_lolv2_synthetic_official_prior",
                  "result": "Diff-Retinex_Plus_lolv2_syn_test", "prefix": None},
}


def rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fid(features_a, features_b):
    mean_a, mean_b = features_a.mean(0), features_b.mean(0)
    cov_a, cov_b = np.cov(features_a, rowvar=False), np.cov(features_b, rowvar=False)
    cov_mean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if not np.isfinite(cov_mean).all():
        offset = np.eye(cov_a.shape[0]) * 1e-6
        cov_mean = linalg.sqrtm((cov_a + offset) @ (cov_b + offset))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    difference = mean_a - mean_b
    return float(difference @ difference + np.trace(cov_a) + np.trace(cov_b)
                 - 2 * np.trace(cov_mean))


@torch.no_grad()
def features(model, images):
    values = []
    for start in range(0, len(images), 10):
        batch = torch.cat([tensor(image) for image in images[start:start + 10]], 0).cuda()
        values.append(model(batch)[0].flatten(1).cpu().numpy())
    return np.concatenate(values, 0)


@torch.no_grad()
def main():
    rchre = RCHRE().cuda().eval().requires_grad_(False)
    rchre.load_state_dict(torch.load(MODEL, map_location="cpu")["params"], strict=True)
    rsra = ReliabilityResidualGate().cuda().eval().requires_grad_(False)
    rsra.load_state_dict(torch.load(RSRA, map_location="cpu")["params"], strict=True)
    inception = InceptionV3([3], resize_input=True, normalize_input=True).cuda().eval()
    locked = json.loads(SOURCE.read_text())
    report = {"protocol": {"diffusion_seeds": [0, 1, 2],
        "aggregation": "300 predictions and repeated paired GT per split",
        "mae": "mean absolute RGB error in [0,1]",
        "fid": "patched InceptionV3, TensorFlow FID weights",
        "source_three_metrics": str(SOURCE.relative_to(ROOT))}, "results": {}}
    for split, cfg in SETS.items():
        methods = ("baseline", "learned_scalar", "rsra", "rchre")
        images = {method: [] for method in methods}
        targets, maes = [], {method: [] for method in methods}
        for seed in (0, 1, 2):
            suffix = "" if seed == 0 else f"_seed{seed}"
            base_dir = ROOT / "results" / (cfg["result"] + suffix) / "visualization"
            for path in sorted(cfg["low"].glob("*.png")):
                gt_name = cfg["prefix"] + path.name[3:] if cfg["prefix"] else path.name
                target_u8 = rgb(cfg["gt"] / gt_name)
                low = tensor(rgb(path)).cuda()
                baseline = tensor(rgb(base_dir / path.name)).cuda()
                prior = tensor(rgb(cfg["prior"] / path.name)).cuda()
                rchre_output, _, _, _ = rchre(low, baseline, prior)
                rsra_output, _ = apply_rsra(rsra, low, baseline, prior)
                predictions = {"baseline": baseline,
                    "learned_scalar": ((1 - ALPHA) * baseline + ALPHA * prior).clamp(0, 1),
                    "rsra": rsra_output, "rchre": rchre_output}
                targets.append(target_u8)
                for method, prediction in predictions.items():
                    image = (prediction[0].permute(1, 2, 0).cpu().numpy()
                             * 255).round().clip(0, 255).astype(np.uint8)
                    images[method].append(image)
                    maes[method].append(float(np.abs(image.astype(np.float64)
                        - target_u8.astype(np.float64)).mean() / 255.))
        target_features = features(inception, targets)
        result = {}
        for method in methods:
            three = locked["results"][split]["means"][method]
            result[method] = {**three, "fid": fid(
                target_features, features(inception, images[method])),
                "mae": float(np.mean(maes[method]))}
        report["results"][split] = result
        print(split, json.dumps(result), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
