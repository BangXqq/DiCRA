#!/usr/bin/env python3
"""One-pass evaluation of pre-registered DiCRA matched-budget controls."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import lpips
import numpy as np
import torch
from basicsr.archs.inception import InceptionV3
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_rchre_lolv1_five_metrics import fid, inception_features
from tools.run_dicra_fair_controls_20261015 import (
    Correction, OUT, VARIANTS, train_split)

DATASETS = {
    "Dev100": {"low": ROOT / "dataset/LOL-v1/our485/low",
               "gt": ROOT / "dataset/LOL-v1/our485/high",
               "prior": ROOT / "experiments/retinexformer_lolv1_prior_train64",
               "baseline": ROOT / "experiments/ddm_train_outputs_128",
               "seed_count": 1},
    "LOLv1": {"low": ROOT / "dataset/LOL-v1/eval15/low",
              "gt": ROOT / "dataset/LOL-v1/eval15/high",
              "prior": ROOT / "experiments/retinexformer_lolv1_official_prior/val",
              "baseline": ROOT / "results/Diff-Retinex_Plus_lolv1_test/visualization",
              "seed_count": 1},
    "Real": {"low": ROOT / "dataset/LOLv2_Real/Test/Low",
             "gt": ROOT / "dataset/LOLv2_Real/Test/Normal",
             "prior": ROOT / "experiments/retinexformer_lolv2_real_official_prior",
             "baseline": ROOT / "results/Diff-Retinex_Plus_lolv2_real_test/visualization",
             "prefix": "normal", "seed_count": 3},
    "Synthetic": {"low": ROOT / "dataset/LOLv2_Synthetic/Test/Low",
                  "gt": ROOT / "dataset/LOLv2_Synthetic/Test/Normal",
                  "prior": ROOT / "experiments/retinexformer_lolv2_synthetic_official_prior",
                  "baseline": ROOT / "results/Diff-Retinex_Plus_lolv2_syn_test/visualization",
                  "seed_count": 3},
}


def rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_tensor(image):
    return torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()


def to_image(tensor):
    return (tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def load_models(alpha):
    models, hashes = {}, {}
    for variant in VARIANTS:
        path = OUT / f"{variant}.pt"
        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["step"] == 4000 and abs(checkpoint["alpha385"] - alpha) < 1e-12
        model = Correction(variant, alpha).cuda().eval().requires_grad_(False)
        model.load_state_dict(checkpoint["params"], strict=True)
        models[variant] = model
        hashes[variant] = hashlib.sha256(path.read_bytes()).hexdigest()
    return models, hashes


def paths_for_split(name, cfg, dev_names):
    if name == "Dev100":
        return [cfg["low"] / filename for filename in dev_names]
    return sorted(cfg["low"].glob("*.png"))


def paired_interval(rows, candidate, reference, metric):
    by_image = {}
    for row_a, row_b in zip(rows[candidate], rows[reference]):
        assert row_a["image"] == row_b["image"] and row_a["seed"] == row_b["seed"]
        by_image.setdefault(row_a["image"], []).append(row_a[metric] - row_b[metric])
    deltas = np.asarray([np.mean(values) for values in by_image.values()])
    rng = np.random.default_rng(20260923)
    samples = rng.integers(0, len(deltas), (20000, len(deltas)))
    boot = deltas[samples].mean(1)
    return {"delta": float(deltas.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "n_images": len(deltas)}


@torch.no_grad()
def evaluate_split(name, cfg, models, perceptual, inception, dev_names):
    methods = ("baseline",) + VARIANTS
    rows = {method: [] for method in methods}
    images = {method: [] for method in methods}
    targets = []
    paths = paths_for_split(name, cfg, dev_names)
    expected = 100 if name != "LOLv1" else 15
    assert len(paths) == expected, (name, len(paths))
    for seed in range(cfg["seed_count"]):
        baseline_dir = cfg["baseline"]
        if seed:
            baseline_dir = Path(str(baseline_dir.parent) + f"_seed{seed}") / "visualization"
        for path in paths:
            target_name = (cfg.get("prefix", "") + path.name[3:]
                           if cfg.get("prefix") else path.name)
            target = rgb(cfg["gt"] / target_name)
            low = to_tensor(rgb(path))
            ddm = to_tensor(rgb(baseline_dir / path.name))
            prior = to_tensor(rgb(cfg["prior"] / path.name))
            assert low.shape == ddm.shape == prior.shape == to_tensor(target).shape
            predictions = {"baseline": ddm}
            predictions.update({variant: model(low, ddm, prior)[0]
                                for variant, model in models.items()})
            target_tensor = to_tensor(target)
            targets.append(target)
            for method, prediction in predictions.items():
                output = to_image(prediction)
                images[method].append(output)
                rows[method].append({"image": path.name, "seed": seed,
                    "psnr": float(calculate_psnr(output, target, crop_border=0)),
                    "ssim": float(calculate_ssim(output, target, crop_border=0)),
                    "lpips": float(perceptual(to_tensor(output) * 2 - 1,
                                               target_tensor * 2 - 1).item()),
                    "mae": float(np.abs(output.astype(np.float64) - target.astype(np.float64)).mean() / 255.)})
        print(f"{name}: seed {seed} paired metrics complete", flush=True)
    target_features = inception_features(inception, targets)
    means = {}
    for method in methods:
        means[method] = {key: float(np.mean([row[key] for row in rows[method]]))
                         for key in ("psnr", "ssim", "lpips", "mae")}
        means[method]["fid"] = fid(target_features, inception_features(inception, images[method]))
    comparisons = {reference: {metric: paired_interval(rows, "dicra", reference, metric)
                               for metric in ("psnr", "ssim", "lpips", "mae")}
                   for reference in ("baseline", "scalar", "single_gate", "no_decomposition")}
    comparisons["isolated_decomposition_no_acceptance"] = {
        metric: paired_interval(rows, "dicra_no_acceptance",
                                "no_decomposition_no_acceptance", metric)
        for metric in ("psnr", "ssim", "lpips", "mae")}
    print(f"{name} means: {json.dumps(means)}", flush=True)
    return {"n_images": len(paths), "seeds": cfg["seed_count"],
            "means": means, "dicra_minus": comparisons, "rows": rows}


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["steps"] == 4000 and protocol["alpha_fit_images"] == 385
    train, dev = train_split()
    assert set(protocol["train_names"]) == {path.name for path in train}
    assert set(protocol["dev_names"]) == {path.name for path in dev}
    models, hashes = load_models(protocol["alpha385"])
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    inception = InceptionV3([3], resize_input=True, normalize_input=True).cuda().eval()
    result = {"protocol": {"alpha_fit": "385 training images only", "alpha385": protocol["alpha385"],
        "seeds": {"Dev100": [0], "LOLv1": [0], "Real": [0, 1, 2], "Synthetic": [0, 1, 2]},
        "fid": "pooled predictions and repeated GT for LOLv2; 2048-d BasicSR Inception",
        "paired_ci": "20,000 image-level bootstrap samples after seed averaging",
        "checkpoint_sha256": hashes}, "results": {}}
    for name, cfg in DATASETS.items():
        result["results"][name] = evaluate_split(name, cfg, models, perceptual,
                                                   inception, protocol["dev_names"])
        (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
