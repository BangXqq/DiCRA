#!/usr/bin/env python3
"""Locked three-training-seed check for existing scalar/no-decomp/DiCRA models.

No architecture selection or checkpoint selection is performed. The historical
20260922 record is reused; only seeds 20260923 and 20260924 are newly evaluated.
"""

import hashlib
import json
from pathlib import Path
import sys

import lpips
import numpy as np
import torch
from basicsr.archs.inception import InceptionV3
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_dicra_fair_controls_20261015 import (
    DATASETS, fid, inception_features, paired_interval, paths_for_split,
    rgb, to_image, to_tensor,
)
from tools.run_dicra_fair_controls_20261015 import Correction


HISTORICAL = ROOT / "reproduction_records/dicra_fair_controls_20261015"
OUT = ROOT / "reproduction_records/dicra_three_seed_controls_20261022"
VARIANTS = ("scalar", "no_decomposition", "dicra")
NEW_SEEDS = (20260923, 20260924)


def load_seed(train_seed):
    folder = OUT / f"seed{train_seed}"
    protocol = json.loads((folder / "protocol.json").read_text())
    if (protocol["train_seed"] != train_seed or protocol["steps"] != 4000
            or protocol["alpha_fit_images"] != 385
            or tuple(protocol["variants"]) != VARIANTS):
        raise AssertionError(f"Unexpected locked protocol: {folder}")
    old = json.loads((HISTORICAL / "protocol.json").read_text())
    if protocol["train_names"] != old["train_names"] or protocol["dev_names"] != old["dev_names"]:
        raise AssertionError("Train/dev split differs from historical control")
    if abs(protocol["alpha385"] - old["alpha385"]) > 1e-12:
        raise AssertionError("Initialization differs from historical control")
    models, hashes = {}, {}
    for variant in VARIANTS:
        path = folder / f"{variant}.pt"
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint["step"] != 4000 or checkpoint["variant"] != variant:
            raise AssertionError(f"Unexpected checkpoint: {path}")
        model = Correction(variant, protocol["alpha385"]).cuda().eval().requires_grad_(False)
        model.load_state_dict(checkpoint["params"], strict=True)
        models[variant] = model
        hashes[variant] = hashlib.sha256(path.read_bytes()).hexdigest()
    return protocol, models, hashes


@torch.no_grad()
def evaluate_split(name, cfg, models, perceptual, inception, dev_names):
    methods = ("baseline",) + VARIANTS
    rows = {method: [] for method in methods}
    images = {method: [] for method in methods}
    targets = []
    paths = paths_for_split(name, cfg, dev_names)
    expected = 15 if name == "LOLv1" else 100
    if len(paths) != expected:
        raise AssertionError((name, len(paths), expected))
    for diffusion_seed in range(cfg["seed_count"]):
        baseline_dir = cfg["baseline"]
        if diffusion_seed:
            baseline_dir = Path(str(baseline_dir.parent)
                                + f"_seed{diffusion_seed}") / "visualization"
        for path in paths:
            target_name = (cfg.get("prefix", "") + path.name[3:]
                           if cfg.get("prefix") else path.name)
            target = rgb(cfg["gt"] / target_name)
            low = to_tensor(rgb(path))
            ddm = to_tensor(rgb(baseline_dir / path.name))
            prior = to_tensor(rgb(cfg["prior"] / path.name))
            target_tensor = to_tensor(target)
            if low.shape != ddm.shape or ddm.shape != prior.shape or prior.shape != target_tensor.shape:
                raise AssertionError(f"Misaligned input: {name}/{path.name}")
            predictions = {"baseline": ddm}
            predictions.update({variant: model(low, ddm, prior)[0]
                                for variant, model in models.items()})
            targets.append(target)
            for method, prediction in predictions.items():
                output = to_image(prediction)
                images[method].append(output)
                rows[method].append({
                    "image": path.name, "seed": diffusion_seed,
                    "psnr": float(calculate_psnr(output, target, crop_border=0)),
                    "ssim": float(calculate_ssim(output, target, crop_border=0)),
                    "lpips": float(perceptual(to_tensor(output) * 2 - 1,
                                              target_tensor * 2 - 1).item()),
                    "mae": float(np.abs(output.astype(np.float64)
                                             - target.astype(np.float64)).mean() / 255.),
                })
        print(f"{name}: diffusion seed {diffusion_seed} complete", flush=True)
    target_features = inception_features(inception, targets)
    means = {}
    for method in methods:
        means[method] = {metric: float(np.mean([row[metric] for row in rows[method]]))
                         for metric in ("psnr", "ssim", "lpips", "mae")}
        means[method]["fid"] = fid(target_features,
                                   inception_features(inception, images[method]))
    differences = {reference: {metric: paired_interval(rows, "dicra", reference, metric)
                               for metric in ("psnr", "ssim", "lpips", "mae")}
                   for reference in ("scalar", "no_decomposition")}
    print(f"{name}: {json.dumps(means, ensure_ascii=False)}", flush=True)
    return {"n_images": len(paths), "diffusion_seeds": cfg["seed_count"],
            "means": means, "dicra_minus": differences, "rows": rows}


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    historical = json.loads((HISTORICAL / "evaluation.json").read_text())
    historical_protocol = json.loads((HISTORICAL / "protocol.json").read_text())
    result = {"scope": "three fixed training seeds; no test-based selection",
              "train_seeds": [20260922, *NEW_SEEDS],
              "variants": list(VARIANTS),
              "historical_seed_source": str((HISTORICAL / "evaluation.json").relative_to(ROOT)),
              "protocol": {"steps": 4000, "split_seed": 20260922,
                           "alpha_fit_images": 385,
                           "alpha385": historical_protocol["alpha385"],
                           "dev100_status": "historical development set, not independent holdout",
                           "fid": "pooled predictions with repeated GT on LOLv2; BasicSR 2048-d Inception; feature batch 5"},
              "results": {"20260922": {
                  "checkpoint_sha256": historical["protocol"]["checkpoint_sha256"],
                  "datasets": {name: historical["results"][name]
                               for name in DATASETS}}}}
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    inception = InceptionV3([3], resize_input=True, normalize_input=True).cuda().eval()
    for train_seed in NEW_SEEDS:
        protocol, models, hashes = load_seed(train_seed)
        entry = {"checkpoint_sha256": hashes, "datasets": {}}
        result["results"][str(train_seed)] = entry
        for name, cfg in DATASETS.items():
            entry["datasets"][name] = evaluate_split(
                name, cfg, models, perceptual, inception, protocol["dev_names"])
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Three-seed evaluation complete", flush=True)


if __name__ == "__main__":
    main()
