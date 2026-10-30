#!/usr/bin/env python3
"""Evaluate LOL-v1 eval15 with PSNR, SSIM, LPIPS, patched-Inception FID and MAE."""
import json
from pathlib import Path
import sys

import cv2
import lpips
import numpy as np
from scipy import linalg
import torch
from basicsr.archs.inception import InceptionV3
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_lsrw_official_checkpoint_fusion import tensor
from tools.train_reliability_residual_gate import (
    ReliabilityResidualGate, apply_gate as apply_rsra, ALPHA)
from tools.train_rchre_validation import RCHRE

LOW = ROOT / "dataset/LOL-v1/eval15/low"
GT = ROOT / "dataset/LOL-v1/eval15/high"
PRIOR = ROOT / "experiments/retinexformer_lolv1_official_prior/val"
BASELINES = {
    "official_checkpoint": ROOT / "results/Diff-Retinex_Plus_lolv1_test/visualization",
    "local_resume170k": ROOT / "results/Diff-Retinex_Plus_lolv1_resume170k_test/visualization",
}
RSRA_MODEL = ROOT / "experiments/reliability_residual_gate_v1/gate.pt"
RCHRE_MODEL = ROOT / "experiments/rchre_validation_v1/model.pt"
OUT = ROOT / "reproduction_records/rchre_lolv1_five_metrics_20260923"


def rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fid(features_a, features_b):
    mean_a, mean_b = features_a.mean(0), features_b.mean(0)
    cov_a = np.cov(features_a, rowvar=False)
    cov_b = np.cov(features_b, rowvar=False)
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
def inception_features(model, images):
    values = []
    for start in range(0, len(images), 5):
        batch = torch.cat([tensor(image) for image in images[start:start + 5]], 0).cuda()
        values.append(model(batch)[0].flatten(1).cpu().numpy())
    return np.concatenate(values, 0)


@torch.no_grad()
def main():
    rsra = ReliabilityResidualGate().cuda().eval().requires_grad_(False)
    rsra.load_state_dict(torch.load(RSRA_MODEL, map_location="cpu")["params"], strict=True)
    rchre = RCHRE().cuda().eval().requires_grad_(False)
    rchre.load_state_dict(torch.load(RCHRE_MODEL, map_location="cpu")["params"], strict=True)
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    inception = InceptionV3([3], resize_input=True, normalize_input=True).cuda().eval()
    paths = sorted(LOW.glob("*.png"))
    assert len(paths) == 15
    targets = [rgb(GT / path.name) for path in paths]
    target_features = inception_features(inception, targets)
    report = {"protocol": {
        "test_set": "LOL-v1 eval15 (15 paired images)",
        "metrics": "RGB full-resolution PSNR/SSIM/LPIPS/MAE; patched-InceptionV3 FID",
        "fid_weights": "pt_inception-2015-12-05-6726825d.pth",
        "diffusion_seed": 0, "module_training": "LOL-v1 our485 only",
        "target_test_fitting": False,
        "official_checkpoint": "pretrained_weights/lolv1/net_g_ddpm_val_lolv1.pth",
        "local_checkpoint": "experiments/diff-retinex_plus_lolv1_resume395k/models/net_g_ddpm_170000.pth"
    }, "results": {}}
    for protocol, base_dir in BASELINES.items():
        rows = {name: [] for name in ("baseline", "learned_scalar", "rsra", "rchre")}
        output_images = {name: [] for name in rows}
        output_root = OUT / protocol
        for path, target_u8 in zip(paths, targets):
            low = tensor(rgb(path)).cuda()
            baseline = tensor(rgb(base_dir / path.name)).cuda()
            prior = tensor(rgb(PRIOR / path.name)).cuda()
            rsra_output, _ = apply_rsra(rsra, low, baseline, prior)
            rchre_output, _, _, _ = rchre(low, baseline, prior)
            predictions = {"baseline": baseline,
                "learned_scalar": ((1 - ALPHA) * baseline + ALPHA * prior).clamp(0, 1),
                "rsra": rsra_output, "rchre": rchre_output}
            target_tensor = tensor(target_u8).cuda()
            for method, prediction in predictions.items():
                image = (prediction[0].permute(1, 2, 0).cpu().numpy()
                         * 255).round().clip(0, 255).astype(np.uint8)
                output_images[method].append(image)
                rows[method].append({"image": path.stem,
                    "psnr": float(calculate_psnr(image, target_u8, crop_border=0)),
                    "ssim": float(calculate_ssim(image, target_u8, crop_border=0)),
                    "lpips": float(perceptual(tensor(image).cuda() * 2 - 1,
                                               target_tensor * 2 - 1).item()),
                    "mae": float(np.abs(image.astype(np.float64)
                                         - target_u8.astype(np.float64)).mean() / 255.)})
                method_dir = output_root / method
                method_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(method_dir / path.name),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = {"means": {}, "rows": rows}
        for method, values in rows.items():
            result["means"][method] = {
                metric: float(np.mean([row[metric] for row in values]))
                for metric in ("psnr", "ssim", "lpips", "mae")}
            result["means"][method]["fid"] = fid(
                target_features, inception_features(inception, output_images[method]))
        report["results"][protocol] = result
        print(protocol, json.dumps(result["means"], ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
