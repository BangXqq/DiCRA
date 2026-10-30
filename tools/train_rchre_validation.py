#!/usr/bin/env python3
"""Pre-registered RCHRE train/validation experiment on LOL-v1 outputs.

RCHRE uses learnable illumination, chroma and detail residual experts,
independent sigmoid routes, oracle route supervision and an explicit fallback
gate to the frozen diffusion anchor.  Target tests are not accessed here.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys

import cv2
import lpips
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_response_prior_adapter as common
from tools import train_spatial_frequency_gate as base
from tools.train_reliability_residual_gate import (
    ReliabilityResidualGate, apply_gate as apply_rsra, ALPHA)

USE_RELIABILITY = os.environ.get("RCHRE_USE_RELIABILITY", "1") == "1"
LEARNABLE_EXPERTS = os.environ.get("RCHRE_LEARNABLE_EXPERTS", "1") == "1"
USE_GLOBAL_ACCEPTANCE = os.environ.get("RCHRE_GLOBAL_ACCEPTANCE", "1") == "1"
USE_SAFETY_LOSS = os.environ.get("RCHRE_USE_SAFETY_LOSS", "1") == "1"
ACTIVE_EXPERTS = tuple(int(value) for value in os.environ.get(
    "RCHRE_ACTIVE_EXPERTS", "1,1,1").split(","))
if len(ACTIVE_EXPERTS) != 3 or any(value not in (0, 1) for value in ACTIVE_EXPERTS):
    raise ValueError("RCHRE_ACTIVE_EXPERTS must contain three comma-separated 0/1 values")
if not any(ACTIVE_EXPERTS):
    raise ValueError("At least one expert must remain active")
ROUTING = os.environ.get("RCHRE_ROUTING", "sigmoid")
if ROUTING not in ("sigmoid", "softmax"):
    raise ValueError("RCHRE_ROUTING must be sigmoid or softmax")
SEED = int(os.environ.get("RCHRE_SEED", "20260922"))
SPLIT_SEED = 20260922
default_out = ("experiments/rchre_validation_v1" if USE_RELIABILITY
               else "experiments/rchre_no_reliability_ablation_v1")
OUT = ROOT / os.environ.get("RCHRE_OUT", default_out)
STEPS, BATCH, CROP = 4000, 4, 128


def split_residual(residual):
    low = F.avg_pool2d(F.pad(residual, (15,) * 4, mode="reflect"), 31, 1)
    # Rec.709 luma; achromatic illumination plus low-frequency chroma sum to low.
    luma = .2126 * low[:, :1] + .7152 * low[:, 1:2] + .0722 * low[:, 2:3]
    illumination = luma.repeat(1, 3, 1, 1)
    chroma = low - illumination
    detail = residual - low
    return illumination, chroma, detail


class LearnableExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 3, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features, component):
        scale = component.abs().mean(1, keepdim=True) + .02
        return component + .25 * torch.tanh(self.body(features)) * scale


class RCHRE(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.experts = nn.ModuleList([LearnableExpert() for _ in range(3)])
        self.router = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 4, 3, padding=1))
        nn.init.zeros_(self.router[-1].weight)
        expert_init = .9
        accept_init = ALPHA / expert_init
        self.router[-1].bias.data.copy_(torch.tensor(
            [math.log(expert_init / (1 - expert_init))] * 3
            + [math.log(accept_init / (1 - accept_init))]))

    def forward(self, low, ddm, prior):
        residual = prior - ddm
        components = split_residual(residual)
        features = self.trunk(torch.cat([low, ddm, prior, residual.abs()], 1))
        experts = ([net(features, component)
                    for net, component in zip(self.experts, components)]
                   if LEARNABLE_EXPERTS else list(components))
        logits = self.router(features)
        expert_routes = (torch.sigmoid(logits[:, :3]) if ROUTING == "sigmoid"
                         else torch.softmax(logits[:, :3], dim=1))
        acceptance = (torch.sigmoid(logits[:, 3:4]) if USE_GLOBAL_ACCEPTANCE
                      else torch.ones_like(logits[:, 3:4]))
        routes = torch.cat([expert_routes, acceptance], 1)
        correction = sum(ACTIVE_EXPERTS[index] * routes[:, index:index + 1] * expert
                         for index, expert in enumerate(experts))
        output = (ddm + routes[:, 3:4] * correction).clamp(0, 1)
        return output, routes, experts, components


def projection(target_minus_ddm, component):
    numerator = (target_minus_ddm * component).sum(1, keepdim=True)
    denominator = component.square().sum(1, keepdim=True)
    oracle = (numerator / denominator.clamp_min(1e-4)).clamp(0, 1)
    oracle = torch.where(denominator > 1e-4, oracle, torch.zeros_like(oracle))
    return F.avg_pool2d(oracle, 5, 1, 2), denominator.detach()


def make_batch(paths, rng):
    values = [[], [], [], []]
    for index in rng.randint(0, len(paths), BATCH):
        sample = base.tensors(paths[int(index)])
        _, h, w = sample[0].shape
        top, left = rng.randint(0, h - CROP + 1), rng.randint(0, w - CROP + 1)
        sample = [x[:, top:top + CROP, left:left + CROP] for x in sample]
        if rng.rand() < .5:
            sample = [x.flip(-1) for x in sample]
        for bucket, value in zip(values, sample):
            bucket.append(value)
    return [torch.stack(items).cuda() for items in values]


def uint_image(value):
    return (value[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def paired(a, b):
    rng = np.random.default_rng(SEED)
    result = {}
    for metric, direction in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
        delta = np.asarray([x[metric] - y[metric] for x, y in zip(a, b)])
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
        result[metric] = {"delta": float(delta.mean()),
                          "ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
                          "wins": int(np.sum(direction * delta > 0))}
    return result


@torch.no_grad()
def evaluate(net, paths, perceptual, rsra):
    rows = {name: [] for name in ("baseline", "learned_scalar", "rsra", "rchre")}
    route_values = []
    expert_diversity = []
    for path in paths:
        low, ddm, prior, target = [x[None].cuda() for x in base.tensors(path)]
        output, routes, experts, _ = net(low, ddm, prior)
        rsra_output, _ = apply_rsra(rsra, low, ddm, prior)
        predictions = {"baseline": ddm,
                       "learned_scalar": ((1 - ALPHA) * ddm + ALPHA * prior).clamp(0, 1),
                       "rsra": rsra_output, "rchre": output}
        target_u8 = uint_image(target)
        for method, prediction in predictions.items():
            image = uint_image(prediction)
            rows[method].append({"image": path.stem,
                "psnr": float(calculate_psnr(image, target_u8, crop_border=0)),
                "ssim": float(calculate_ssim(image, target_u8, crop_border=0)),
                "lpips": float(perceptual(prediction * 2 - 1, target * 2 - 1).item())})
        route_values.append(routes.mean((0, 2, 3)).cpu().tolist())
        expert_diversity.append([
            float(F.cosine_similarity(experts[i].flatten(1), experts[j].flatten(1)).mean())
            for i, j in ((0, 1), (0, 2), (1, 2))])
    means = {method: {metric: float(np.mean([row[metric] for row in values]))
                      for metric in ("psnr", "ssim", "lpips")}
             for method, values in rows.items()}
    return {"means": means,
            "comparisons": {f"rchre-minus-{method}": paired(rows["rchre"], rows[method])
                            for method in ("baseline", "learned_scalar", "rsra")},
            "route_means": np.mean(route_values, 0).tolist(),
            "route_std_across_images": np.std(route_values, 0).tolist(),
            "expert_cosine_means": np.mean(expert_diversity, 0).tolist(),
            "rows": rows}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    paths = np.asarray(sorted(base.LOW.glob("*.png")), dtype=object)
    permutation = np.random.RandomState(SPLIT_SEED).permutation(len(paths))
    train = paths[permutation[:385]].tolist()
    heldout = paths[permutation[385:]].tolist()
    net = RCHRE().cuda().train()
    rsra = ReliabilityResidualGate().cuda().eval().requires_grad_(False)
    rsra.load_state_dict(torch.load(
        ROOT / "experiments/reliability_residual_gate_v1/gate.pt",
        map_location="cpu")["params"], strict=True)
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.RandomState(SEED)
    history = []
    for step in range(1, STEPS + 1):
        low, ddm, prior, target = make_batch(train, rng)
        output, routes, experts, components = net(low, ddm, prior)
        pixel = F.l1_loss(output, target)
        # Use the already validated differentiable implementation.
        from tools.train_spatial_frequency_gate_pareto import ssim_loss
        structure = ssim_loss(output, target)
        perception = perceptual(output * 2 - 1, target * 2 - 1).mean()
        target_residual = target - ddm
        reliability = 0
        expert_oracles, expert_weights = [], []
        for component in components:
            oracle, energy = projection(target_residual, component)
            weight = (energy / (energy.mean().detach() + 1e-6)).clamp(0, 4)
            expert_oracles.append(oracle)
            expert_weights.append(weight)
        if ROUTING == "softmax":
            stacked = torch.cat(expert_oracles, 1)
            total = stacked.sum(1, keepdim=True)
            stacked = torch.where(
                total > 1e-4, stacked / total.clamp_min(1e-4),
                torch.full_like(stacked, 1 / 3))
            expert_oracles = list(stacked.split(1, 1))
        for index, (oracle, weight) in enumerate(zip(
                expert_oracles, expert_weights)):
            if ACTIVE_EXPERTS[index]:
                reliability = reliability + (F.smooth_l1_loss(
                    routes[:, index:index + 1], oracle, reduction="none") * weight).mean()
        routed = sum(ACTIVE_EXPERTS[index] * routes[:, index:index + 1] * experts[index]
                     for index in range(3)).detach()
        if USE_GLOBAL_ACCEPTANCE:
            accept_oracle, accept_energy = projection(target_residual, routed)
            accept_weight = (accept_energy / (accept_energy.mean().detach() + 1e-6)).clamp(0, 4)
            reliability = reliability + (F.smooth_l1_loss(
                routes[:, 3:4], accept_oracle, reduction="none") * accept_weight).mean()
        base_error = (ddm - target).abs().mean((1, 2, 3))
        output_error = (output - target).abs().mean((1, 2, 3))
        safe = F.relu(output_error - base_error).mean()
        tv = sum((routes[:, i:i+1, 1:] - routes[:, i:i+1, :-1]).abs().mean()
                 + (routes[:, i:i+1, :, 1:] - routes[:, i:i+1, :, :-1]).abs().mean()
                 for i in range(4))
        reliability_weight = .05 if USE_RELIABILITY else 0.
        safety_weight = .1 if USE_SAFETY_LOSS else 0.
        loss = (pixel + .2 * structure + .05 * perception
                + reliability_weight * reliability + safety_weight * safe
                + .001 * tv)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            row = {"step": step, "loss": float(loss), "pixel": float(pixel),
                   "reliability": float(reliability), "safe": float(safe),
                   "routes": routes.mean((0, 2, 3)).detach().cpu().tolist()}
            history.append(row)
            print(json.dumps(row), flush=True)
    net.eval()
    validation = evaluate(net, heldout, perceptual, rsra)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"params": {k: v.cpu() for k, v in net.state_dict().items()},
                "step": STEPS, "seed": SEED}, OUT / "model.pt")
    report = {"protocol": {"seed": SEED, "split_seed": SPLIT_SEED,
                            "steps": STEPS,
                            "train_images": 385, "heldout_images": 100,
                            "split": "fixed random permutation before training",
                            "target_test_access": False,
                            "reliability_supervision": USE_RELIABILITY,
                            "learnable_experts": LEARNABLE_EXPERTS,
                            "global_acceptance": USE_GLOBAL_ACCEPTANCE,
                            "safety_loss": USE_SAFETY_LOSS,
                            "active_experts": list(ACTIVE_EXPERTS),
                            "routing": ROUTING,
                            "success_rule": "all means beat baseline; >=1 metric beats scalar and RSRA"},
              "parameters": sum(p.numel() for p in net.parameters()),
              "validation": validation, "history": history}
    common.write_json(OUT / "summary.json", report)
    compact = {k: v for k, v in validation.items() if k not in ("rows",)}
    print(json.dumps({"event": "complete", "parameters": report["parameters"],
                      "validation": compact}), flush=True)


if __name__ == "__main__":
    with open("/tmp/diff-retinex-gpu-0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
