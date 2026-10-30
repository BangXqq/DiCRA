#!/usr/bin/env python3
"""One-shot reliability-supervised residual-acceptance gate experiment.

The router mixes a no-op and a full cross-model correction expert.  In
addition to reconstruction losses, it learns the per-pixel least-squares
acceptance target derived only from LOL-v1 training ground truth.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_response_prior_adapter as common
from tools import train_spatial_frequency_gate as base
from tools import train_spatial_frequency_gate_pareto as pareto

STEPS, BATCH = 4000, 4
SEED = int(os.environ.get("RSRA_TRAIN_SEED", "20260922"))
OUT = ROOT / ("experiments/reliability_residual_gate_v1" if SEED == 20260922
              else f"experiments/reliability_residual_gate_seed{SEED}")
ALPHA = 0.34773870423631975


class ReliabilityResidualGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight)
        self.body[-1].bias.data.fill_(math.log(ALPHA / (1 - ALPHA)))

    def forward(self, low, ddm, prior):
        return torch.sigmoid(self.body(torch.cat(
            [low, ddm, prior, (prior - ddm).abs()], 1)))


def apply_gate(net, low, ddm, prior):
    gate = net(low, ddm, prior)
    return (ddm + gate * (prior - ddm)).clamp(0, 1), gate


def oracle_acceptance(ddm, prior, target):
    """RGB least-squares projection of target-D onto P-D at each pixel."""
    residual = prior - ddm
    numerator = ((target - ddm) * residual).sum(1, keepdim=True)
    denominator = residual.square().sum(1, keepdim=True)
    oracle = (numerator / denominator.clamp_min(1e-4)).clamp(0, 1)
    # Undefined where the two restorers agree; retain the trained scalar there.
    oracle = torch.where(denominator > 1e-4, oracle,
                         torch.full_like(oracle, ALPHA))
    return F.avg_pool2d(oracle, 5, 1, 2), denominator.detach()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    paths = sorted(base.LOW.glob("*.png"))
    assert len(paths) == 485
    net = ReliabilityResidualGate().cuda().train()
    import lpips
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.RandomState(SEED)
    history = []
    for step in range(1, STEPS + 1):
        low, ddm, prior, target = pareto.make_batch(paths, rng)
        output, gate = apply_gate(net, low, ddm, prior)
        oracle, energy = oracle_acceptance(ddm, prior, target)
        pixel = F.l1_loss(output, target)
        structure = pareto.ssim_loss(output, target)
        perception = perceptual(output * 2 - 1, target * 2 - 1).mean()
        confidence = (energy / (energy.mean().detach() + 1e-6)).clamp(0, 4)
        reliability = (F.smooth_l1_loss(gate, oracle, reduction="none") * confidence).mean()
        tv = ((gate[..., 1:, :] - gate[..., :-1, :]).abs().mean()
              + (gate[..., :, 1:] - gate[..., :, :-1]).abs().mean())
        # Coefficients fixed before target-dataset evaluation.
        loss = pixel + .2 * structure + .05 * perception + .1 * reliability + .002 * tv
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            row = {"step": step, "loss": float(loss), "pixel": float(pixel),
                   "ssim_loss": float(structure), "lpips": float(perception),
                   "reliability": float(reliability), "gate": float(gate.mean()),
                   "oracle": float(oracle.mean())}
            history.append(row)
            print(json.dumps(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"params": {k: v.cpu() for k, v in net.state_dict().items()},
                "step": STEPS, "alpha_init": ALPHA,
                "protocol": {"train_images": 485, "seed": SEED,
                             "test_tuning": False,
                             "loss": "L1+.2DSSIM+.05LPIPS+.1Reliability+.002TV"}},
               OUT / "gate.pt")
    common.write_json(OUT / "train.json", {
        "steps": STEPS, "train_images": 485, "seed": SEED,
        "parameters": sum(p.numel() for p in net.parameters()),
        "alpha_init": ALPHA, "history": history})
    print(json.dumps({"event": "complete",
                      "parameters": sum(p.numel() for p in net.parameters())}), flush=True)


if __name__ == "__main__":
    with open("/tmp/diff-retinex-gpu-0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
