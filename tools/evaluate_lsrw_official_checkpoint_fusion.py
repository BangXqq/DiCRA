#!/usr/bin/env python3
"""Evaluate frozen LOL-v1 prior fusion on LSRW-checkpoint outputs.

No LSRW labels are used for fitting or model selection. The simple average and
the spatial gate are the candidates frozen before this evaluation.
"""
import json
from pathlib import Path
import sys

import cv2
import lpips
import numpy as np
import torch
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_spatial_frequency_gate import SpatialGate, apply_gate

DATA = ROOT / "dataset/LSRW_640x480/Eval"
PRIOR = ROOT / "experiments/retinexformer_lolv1_prior_lsrw_640x480"
GATE = ROOT / "experiments/spatial_frequency_gate_pareto_full_v1/gate.pt"
OUT = ROOT / "reproduction_records/lsrw_640x480_checkpoint_fusion_20260920"
BASE = {
    "Nikon": ROOT / "results/Diff-Retinex_Plus_LSRW-Nikon_test/visualization",
    "Huawei": ROOT / "results/Diff-Retinex_Plus_LSRW_Huawei_test/visualization",
}


def rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def tensor(im):
    return torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1)[None]


def comparison(a, b, seed=20260920, samples=10000):
    rng = np.random.default_rng(seed)
    ans = {}
    for key, better in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
        d = np.asarray([x[key] - y[key] for x, y in zip(a, b)])
        means = d[rng.integers(0, len(d), (samples, len(d)))].mean(1)
        ans[key] = {"delta": float(d.mean()), "ci95": [float(x) for x in np.quantile(means, [.025, .975])],
                    "wins": int(np.sum(better * d > 0)), "n": len(d)}
    return ans


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gate = SpatialGate().cuda().eval().requires_grad_(False)
    gate.load_state_dict(torch.load(GATE, map_location="cpu")["params"])
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    report = {"protocol": {"lsrw_training": "none", "candidate_selection": "LOL-v1 only",
                            "baseline_checkpoint": "net_g_ddpm_val_lsrw.pth",
                            "prior_checkpoint": "RetinexFormer LOL_v1.pth"}, "results": {}}
    for camera in ("Nikon", "Huawei"):
        rows = {x: [] for x in ("baseline", "prior", "simple_average", "spatial_gate")}
        for gt_path in sorted((DATA / camera / "high").glob("*.jpg")):
            name = gt_path.stem
            images = {
                "baseline": rgb(BASE[camera] / f"{name}.png"),
                "prior": rgb(PRIOR / camera.lower() / f"{name}.png"),
            }
            gt = rgb(gt_path)
            low = tensor(rgb(DATA / camera / "low" / gt_path.name)).cuda()
            base, prior = tensor(images["baseline"]).cuda(), tensor(images["prior"]).cuda()
            average = (base + prior) / 2
            spatial, _ = apply_gate(gate, low, base, prior)
            images["simple_average"] = (average[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            images["spatial_gate"] = (spatial[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            gt_t = tensor(gt).cuda()
            for method, im in images.items():
                pred = tensor(im).cuda()
                rows[method].append({"image": name,
                    "psnr": float(calculate_psnr(im, gt, crop_border=0)),
                    "ssim": float(calculate_ssim(im, gt, crop_border=0)),
                    "lpips": float(perceptual(pred * 2 - 1, gt_t * 2 - 1).item())})
        means = {m: {k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ssim", "lpips")} for m, rs in rows.items()}
        report["results"][camera] = {"means": means, "comparisons": {
            "average-minus-baseline": comparison(rows["simple_average"], rows["baseline"]),
            "spatial-minus-baseline": comparison(rows["spatial_gate"], rows["baseline"]),
            "spatial-minus-average": comparison(rows["spatial_gate"], rows["simple_average"]),
        }, "rows": rows}
        print(camera, json.dumps(means), flush=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
