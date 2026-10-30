#!/usr/bin/env python3
"""Evaluate frozen RCHRE on LSRW 640x480 with PSNR/SSIM/LPIPS/FID/MAE."""
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
from tools.evaluate_lsrw_official_checkpoint_fusion import tensor, comparison
from tools.train_rchre_validation import RCHRE

DATA = ROOT / "dataset/LSRW_640x480/Eval"
PRIOR = ROOT / "experiments/retinexformer_lolv1_prior_lsrw_640x480"
BASE = {
    "Nikon": ROOT / "results/Diff-Retinex_Plus_LSRW-Nikon_test/visualization",
    "Huawei": ROOT / "results/Diff-Retinex_Plus_LSRW_Huawei_test/visualization",
}
MODEL = ROOT / "experiments/rchre_validation_v1/model.pt"
OUT = ROOT / "reproduction_records/rchre_lsrw_five_metrics_20260923"


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
def inception_features(model, images):
    values = []
    for start in range(0, len(images), 5):
        batch = torch.cat([tensor(image) for image in images[start:start + 5]], 0).cuda()
        values.append(model(batch)[0].flatten(1).cpu().numpy())
    return np.concatenate(values, 0)


@torch.no_grad()
def main():
    net = RCHRE().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(MODEL, map_location="cpu")["params"], strict=True)
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    inception = InceptionV3([3], resize_input=True, normalize_input=True).cuda().eval()
    report = {"protocol": {"resolution": "640x480", "lsrw_training": "none",
        "module_training": "LOL-v1 only", "training_seed": 20260922,
        "baseline_checkpoint": "net_g_ddpm_val_lsrw.pth",
        "prior_checkpoint": "RetinexFormer LOL_v1.pth",
        "metrics": "RGB full-resolution PSNR/SSIM/LPIPS/MAE; patched-InceptionV3 FID",
        "misregistration_warning": True}, "results": {}}
    for camera in ("Nikon", "Huawei"):
        rows = {method: [] for method in ("baseline", "rchre")}
        images = {method: [] for method in rows}
        targets = []
        routes = []
        for gt_path in sorted((DATA / camera / "high").glob("*.jpg")):
            name = gt_path.stem
            target_u8 = rgb(gt_path)
            targets.append(target_u8)
            target = tensor(target_u8).cuda()
            low = tensor(rgb(DATA / camera / "low" / gt_path.name)).cuda()
            baseline = tensor(rgb(BASE[camera] / f"{name}.png")).cuda()
            prior = tensor(rgb(PRIOR / camera.lower() / f"{name}.png")).cuda()
            output, route, _, _ = net(low, baseline, prior)
            routes.append(route.mean((0, 2, 3)).cpu().tolist())
            for method, prediction in {"baseline": baseline, "rchre": output}.items():
                image = (prediction[0].permute(1, 2, 0).cpu().numpy()
                         * 255).round().clip(0, 255).astype(np.uint8)
                images[method].append(image)
                rows[method].append({"image": name,
                    "psnr": float(calculate_psnr(image, target_u8, crop_border=0)),
                    "ssim": float(calculate_ssim(image, target_u8, crop_border=0)),
                    "lpips": float(perceptual(tensor(image).cuda() * 2 - 1,
                                               target * 2 - 1).item()),
                    "mae": float(np.abs(image.astype(np.float64)
                                         - target_u8.astype(np.float64)).mean() / 255.)})
        target_features = inception_features(inception, targets)
        means = {}
        for method, values in rows.items():
            means[method] = {metric: float(np.mean([row[metric] for row in values]))
                             for metric in ("psnr", "ssim", "lpips", "mae")}
            means[method]["fid"] = fid(
                target_features, inception_features(inception, images[method]))
        result = {"means": means,
            "rchre_minus_baseline": comparison(rows["rchre"], rows["baseline"]),
            "route_means": np.mean(routes, 0).tolist(), "rows": rows}
        report["results"][camera] = result
        print(camera, json.dumps({"means": means,
            "rchre_minus_baseline": result["rchre_minus_baseline"]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
