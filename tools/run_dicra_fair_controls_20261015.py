#!/usr/bin/env python3
"""Locked, train-only calibration and matched-budget DiCRA controls.

Run all variants before inspecting any official test results.  The common
reconstruction objective, cached backbones, crop stream and 4000-step budget
are fixed.  Gate-specific oracle losses are documented, not silently treated
as identical losses across different architectures.
"""
import argparse
import cv2
import fcntl
import json
import math
import os
import random
from pathlib import Path
import sys

import lpips
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_spatial_frequency_gate as data
from tools.train_rchre_validation import LearnableExpert, projection, split_residual
from tools.train_spatial_frequency_gate_pareto import ssim_loss

OUT = ROOT / os.environ.get(
    "DICRA_FAIR_OUT", "reproduction_records/dicra_fair_controls_20261015")
SPLIT_SEED = 20260922
TRAIN_SEED = int(os.environ.get("DICRA_FAIR_TRAIN_SEED", "20260922"))
STEPS = 4000
BATCH = 4
CROP = 128
VARIANTS = ("scalar", "single_gate", "no_decomposition", "dicra",
            "no_decomposition_no_acceptance", "dicra_no_acceptance")


def train_split():
    paths = np.asarray(sorted(data.LOW.glob("*.png")), dtype=object)
    if len(paths) != 485:
        raise RuntimeError(f"Expected 485 LOL-v1 training pairs, found {len(paths)}")
    order = np.random.RandomState(SPLIT_SEED).permutation(485)
    return paths[order[:385]].tolist(), paths[order[385:]].tolist()


def fit_alpha(paths):
    numerator = 0.0
    denominator = 0.0
    for path in paths:
        _, ddm, prior, target = data.tensors(path)
        residual = (prior - ddm).double()
        numerator += ((target - ddm).double() * residual).sum().item()
        denominator += residual.square().sum().item()
    return min(1.0, max(0.0, numerator / denominator))


def load_training_cache(paths):
    """Keep the four 8-bit RGB sources in RAM; retain the original input values."""
    roots = (data.LOW, data.DDM, data.RF, data.GT)
    cache = []
    for path in paths:
        sample = []
        for root in roots:
            image = cv2.imread(str(root / path.name), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(root / path.name)
            sample.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if len({image.shape for image in sample}) != 1:
            raise ValueError(f"Misaligned cached sources: {path.name}")
        cache.append(sample)
    return cache


def logit(value):
    return math.log(value / (1.0 - value))


class Correction(nn.Module):
    def __init__(self, variant, alpha):
        super().__init__()
        self.variant = variant
        if variant == "scalar":
            self.scalar_logit = nn.Parameter(torch.tensor(logit(alpha)))
            return
        if variant == "single_gate":
            self.gate = nn.Sequential(
                nn.Conv2d(12, 32, 3, padding=1), nn.SiLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
                nn.Conv2d(16, 1, 3, padding=1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, logit(alpha))
            return
        if variant not in ("no_decomposition", "dicra",
                           "no_decomposition_no_acceptance", "dicra_no_acceptance"):
            raise ValueError(variant)
        self.trunk = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.experts = nn.ModuleList(LearnableExpert() for _ in range(3))
        self.router = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 4, 3, padding=1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias[:3], logit(.9))
        nn.init.constant_(self.router[-1].bias[3:], logit(alpha / .9))

    def forward(self, low, ddm, prior):
        residual = prior - ddm
        if self.variant == "scalar":
            gate = torch.sigmoid(self.scalar_logit)
            return (ddm + gate * residual).clamp(0, 1), None, None, None
        x = torch.cat((low, ddm, prior, residual.abs()), dim=1)
        if self.variant == "single_gate":
            gate = torch.sigmoid(self.gate(x))
            return (ddm + gate * residual).clamp(0, 1), gate, [residual], residual
        components = (split_residual(residual) if self.variant.startswith("dicra")
                      else (residual / 3, residual / 3, residual / 3))
        features = self.trunk(x)
        experts = [net(features, component)
                   for net, component in zip(self.experts, components)]
        gates = torch.sigmoid(self.router(features))
        combined = sum(gates[:, i:i+1] * experts[i] for i in range(3))
        output = (ddm + gates[:, 3:4] * combined).clamp(0, 1)
        return output, gates, components, combined


def batch(cache, rng, device):
    fields = [[], [], [], []]
    for index in rng.randint(0, len(cache), BATCH):
        sample = cache[int(index)]
        height, width = sample[0].shape[:2]
        top = rng.randint(0, height - CROP + 1)
        left = rng.randint(0, width - CROP + 1)
        sample = [item[top:top+CROP, left:left+CROP] for item in sample]
        if rng.rand() < .5:
            sample = [item[:, ::-1] for item in sample]
        for field, value in zip(fields, sample):
            field.append(torch.from_numpy(value.copy().transpose(2, 0, 1)).float() / 255.)
    return [torch.stack(field).to(device) for field in fields]


def weighted_projection_loss(predicted, target_minus_ddm, component):
    oracle, energy = projection(target_minus_ddm, component.detach())
    weight = (energy / (energy.mean() + 1e-6)).clamp(0, 4)
    return (F.smooth_l1_loss(predicted, oracle, reduction="none") * weight).mean()


def train_one(variant, alpha, cache, device):
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    model = Correction(variant, alpha).to(device).train()
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    perceptual.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.RandomState(TRAIN_SEED)
    history = []
    for step in range(1, STEPS + 1):
        low, ddm, prior, target = batch(cache, rng, device)
        output, gates, components, combined = model(low, ddm, prior)
        pixel = F.l1_loss(output, target)
        structure = ssim_loss(output, target)
        perception = perceptual(output * 2 - 1, target * 2 - 1).mean()
        base_error = (ddm - target).abs().mean((1, 2, 3))
        output_error = (output - target).abs().mean((1, 2, 3))
        anchor = F.relu(output_error - base_error).mean()
        acceptance = output.new_zeros(())
        tv = output.new_zeros(())
        if gates is not None:
            target_residual = target - ddm
            if variant.endswith("no_acceptance"):
                pass
            elif variant == "single_gate":
                acceptance = weighted_projection_loss(gates, target_residual, components[0])
            else:
                acceptance = sum(weighted_projection_loss(
                    gates[:, index:index+1], target_residual, component)
                    for index, component in enumerate(components))
                acceptance = acceptance + weighted_projection_loss(
                    gates[:, 3:4], target_residual, combined)
            tv = ((gates[..., 1:, :] - gates[..., :-1, :]).abs().mean()
                  + (gates[..., :, 1:] - gates[..., :, :-1]).abs().mean())
        loss = pixel + .2 * structure + .05 * perception + .1 * anchor
        loss = loss + .05 * acceptance + .001 * tv
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == 1:
            record = {"variant": variant, "step": step, "loss": float(loss.detach()),
                      "pixel": float(pixel.detach()), "acceptance": float(acceptance.detach())}
            history.append(record)
            print(json.dumps(record), flush=True)
    checkpoint = OUT / f"{variant}.pt"
    torch.save({"params": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "variant": variant, "alpha385": alpha, "step": STEPS}, checkpoint)
    return {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint": str(checkpoint.relative_to(ROOT)), "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--pilot-steps", type=int, default=0)
    args = parser.parse_args()
    global STEPS
    if args.pilot_steps:
        STEPS = args.pilot_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full matched-budget training requires CUDA")
    torch.set_num_threads(4)
    train, dev = train_split()
    alpha = fit_alpha(train)
    cache = load_training_cache(train)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"split_seed": SPLIT_SEED, "train_seed": TRAIN_SEED,
                "train_images": len(train), "dev_images": len(dev),
                "alpha_fit_images": len(train), "alpha385": alpha,
                "steps": STEPS, "batch": BATCH, "crop": CROP,
                "variants": list(args.variants), "official_test_used_during_this_run": False,
                "shared_loss": "L1 + .2*(1-SSIM) + .05*LPIPS + .1*anchor-MAE",
                "gate_loss": ".05*energy-weighted projection SmoothL1 + .001*TV; absent for scalar",
                "no_decomposition": "three R/3 branches; exactly same topology and parameters as DiCRA",
                "isolation_pair": "dicra_no_acceptance vs no_decomposition_no_acceptance; identical loss/topology, while component-dependent expert scaling follows each representation",
                "train_names": [p.name for p in train], "dev_names": [p.name for p in dev]}
    existing_path = OUT / "protocol.json"
    results = []
    if existing_path.exists() and not args.pilot_steps:
        existing = json.loads(existing_path.read_text())
        if (existing["steps"] == STEPS and existing["alpha385"] == alpha
                and existing["train_names"] == protocol["train_names"]):
            protocol["variants"] = list(dict.fromkeys(existing["variants"] + list(args.variants)))
            summary_path = OUT / "training_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["protocol"]["steps"] == STEPS:
                    results = summary["results"]
    existing_path.write_text(json.dumps(protocol, indent=2) + "\n")
    for variant in args.variants:
        results.append(train_one(variant, alpha, cache, device))
        (OUT / "training_summary.json").write_text(json.dumps(
            {"protocol": protocol, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    with open("/tmp/diff-retinex-gpu-0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
