#!/usr/bin/env python3
"""Single locked LOLv2 test of validation-qualified RCHRE."""
import json
import os
from pathlib import Path
import sys

import cv2
import lpips
import numpy as np
import torch
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_lsrw_official_checkpoint_fusion import tensor
from tools.train_reliability_residual_gate import ReliabilityResidualGate, apply_gate as apply_rsra, ALPHA
from tools.train_rchre_validation import RCHRE

OUT = ROOT / os.environ.get(
    "RCHRE_EVAL_OUT", "reproduction_records/rchre_lolv2_locked_20260922")
MODEL = ROOT / os.environ.get(
    "RCHRE_MODEL", "experiments/rchre_validation_v1/model.pt")
DIFFUSION_SEEDS = tuple(int(value) for value in os.environ.get(
    "RCHRE_DIFFUSION_SEEDS", "0,1,2").split(","))
RSRA = ROOT / "experiments/reliability_residual_gate_v1/gate.pt"
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


def means(rows):
    return {method: {metric: float(np.mean([row[metric] for row in values]))
                     for metric in ("psnr", "ssim", "lpips")}
            for method, values in rows.items()}


def image_cluster_comparison(a, b):
    """Bootstrap images after averaging the configured diffusion seeds."""
    rng = np.random.default_rng(20260920)
    result = {}
    names = sorted({row["image"] for row in a})
    for metric, direction in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
        av = {(row["image"], row["seed"]): row[metric] for row in a}
        bv = {(row["image"], row["seed"]): row[metric] for row in b}
        delta = np.asarray([np.mean([
            av[(name, seed)] - bv[(name, seed)] for seed in DIFFUSION_SEEDS
        ]) for name in names])
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
        result[metric] = {
            "delta": float(delta.mean()),
            "ci95": [float(value) for value in np.quantile(boot, [.025, .975])],
            "image_wins": int(np.sum(direction * delta > 0)),
            "n_images": len(delta), "n_seeds": len(DIFFUSION_SEEDS)}
    return result


@torch.no_grad()
def main():
    model = RCHRE().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(MODEL, map_location="cpu")["params"], strict=True)
    rsra = ReliabilityResidualGate().cuda().eval().requires_grad_(False)
    rsra.load_state_dict(torch.load(RSRA, map_location="cpu")["params"], strict=True)
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    report = {"protocol": {"model_selection": "LOL-v1 heldout only",
                           "target_test_tuning": False,
                           "diffusion_seeds": list(DIFFUSION_SEEDS),
                           "checkpoint": str(MODEL.relative_to(ROOT))},
              "parameters": sum(p.numel() for p in model.parameters()), "results": {}}
    for split, cfg in SETS.items():
        rows = {name: [] for name in
                ("baseline", "learned_scalar", "rsra", "rchre")}
        route_values = []
        expert_cosines = []
        for seed in DIFFUSION_SEEDS:
            suffix = "" if seed == 0 else f"_seed{seed}"
            base_dir = ROOT / "results" / (cfg["result"] + suffix) / "visualization"
            for path in sorted(cfg["low"].glob("*.png")):
                gt_name = cfg["prefix"] + path.name[3:] if cfg["prefix"] else path.name
                target_u8 = rgb(cfg["gt"] / gt_name)
                target = tensor(target_u8).cuda()
                low = tensor(rgb(path)).cuda()
                baseline = tensor(rgb(base_dir / path.name)).cuda()
                prior = tensor(rgb(cfg["prior"] / path.name)).cuda()
                rchre, routes, experts, _ = model(low, baseline, prior)
                rsra_output, _ = apply_rsra(rsra, low, baseline, prior)
                predictions = {"baseline": baseline,
                    "learned_scalar": ((1 - ALPHA) * baseline + ALPHA * prior).clamp(0, 1),
                    "rsra": rsra_output, "rchre": rchre}
                route_values.append(routes.mean((0, 2, 3)).cpu().tolist())
                expert_cosines.append([
                    float(torch.nn.functional.cosine_similarity(
                        experts[i].flatten(1), experts[j].flatten(1)).mean())
                    for i, j in ((0, 1), (0, 2), (1, 2))])
                for method, prediction in predictions.items():
                    image = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    rows[method].append({"image": path.stem, "seed": seed,
                        "psnr": float(calculate_psnr(image, target_u8, crop_border=0)),
                        "ssim": float(calculate_ssim(image, target_u8, crop_border=0)),
                        "lpips": float(perceptual(tensor(image).cuda() * 2 - 1,
                                                   target * 2 - 1).item())})
            print(json.dumps({"event": "eval", "split": split, "seed": seed}), flush=True)
        result = {"means": means(rows),
                  "comparisons": {f"rchre-minus-{method}": image_cluster_comparison(
                      rows["rchre"], rows[method])
                      for method in ("baseline", "learned_scalar", "rsra")},
                  "route_means": np.mean(route_values, 0).tolist(),
                  "route_std": np.std(route_values, 0).tolist(),
                  "expert_cosine_means": np.mean(expert_cosines, 0).tolist(),
                  "rows": rows}
        report["results"][split] = result
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    compact = {split: {k: v for k, v in value.items() if k != "rows"}
               for split, value in report["results"].items()}
    print(json.dumps({"event": "complete", "results": compact}), flush=True)


if __name__ == "__main__":
    main()
