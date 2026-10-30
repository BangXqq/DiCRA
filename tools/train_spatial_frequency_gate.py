#!/usr/bin/env python3
"""Feasibility test for a learned spatial low/high restoration-prior gate."""
import fcntl
import json
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.utils import hiseq_color_cv2_img
from tools import run_response_prior_adapter as common

LOW = ROOT / "dataset/LOL-v1/our485/low"
GT = ROOT / "dataset/LOL-v1/our485/high"
DDM = ROOT / "experiments/ddm_train_outputs_128"
RF = ROOT / "experiments/retinexformer_lolv1_prior_train64"
OUT = ROOT / "experiments/spatial_frequency_gate_feasibility_v1"
STEPS, BATCH, CROP, SEED = 4000, 8, 128, 20260920
LOW_INIT, HIGH_INIT = 0.31117931505162066, 0.11345654089129104


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 2, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight)
        self.body[-1].bias.data.copy_(torch.tensor([
            math.log(LOW_INIT / (1 - LOW_INIT)), math.log(HIGH_INIT / (1 - HIGH_INIT))]))

    def forward(self, low, ddm, prior):
        return torch.sigmoid(self.body(torch.cat([low, ddm, prior, (prior-ddm).abs()], 1)))


def read(path):
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.


def tensors(path):
    return tuple(torch.from_numpy(read(root / path.name).transpose(2, 0, 1)).float()
                 for root in (LOW, DDM, RF, GT))


def apply_gate(net, low, ddm, prior):
    residual = prior - ddm
    radius = 15
    low_r = F.avg_pool2d(F.pad(residual, (radius,) * 4, mode="reflect"), 31, 1)
    high_r = residual - low_r
    gates = net(low, ddm, prior)
    return (ddm + gates[:, :1] * low_r + gates[:, 1:] * high_r).clamp(0, 1), gates


def make_batch(paths, rng):
    values = [[], [], [], []]
    for index in rng.randint(0, len(paths), BATCH):
        sample = tensors(paths[int(index)])
        top, left = rng.randint(0, 273), rng.randint(0, 473)
        sample = [x[:, top:top+CROP, left:left+CROP] for x in sample]
        if rng.rand() < .5:
            sample = [x.flip(-1) for x in sample]
        for bucket, value in zip(values, sample): bucket.append(value)
    return [torch.stack(x).cuda() for x in values]


@torch.no_grad()
def evaluate(net, paths, perceptual):
    rows = {m: [] for m in ("ddm", "average", "global_gate", "spatial_gate")}
    gate_values = []
    for path in paths:
        low, ddm, prior, target = [x[None].cuda() for x in tensors(path)]
        spatial, gates = apply_gate(net, low, ddm, prior)
        residual = prior - ddm; radius = 15
        low_r = F.avg_pool2d(F.pad(residual, (radius,) * 4, mode="reflect"), 31, 1)
        global_y = (ddm + LOW_INIT*low_r + HIGH_INIT*(residual-low_r)).clamp(0, 1)
        outputs = {"ddm": ddm, "average": (ddm+prior)/2,
                   "global_gate": global_y, "spatial_gate": spatial}
        gate_values.append([float(gates[:, 0].mean()), float(gates[:, 1].mean())])
        target_u8 = common.hist.uint_image(target)
        for mode, output in outputs.items():
            image = common.hist.uint_image(output)
            rows[mode].append({"image": path.stem,
                "psnr": float(calculate_psnr(image, target_u8, crop_border=0)),
                "ssim": float(calculate_ssim(image, target_u8, crop_border=0)),
                "lpips": float(perceptual(output*2-1, target*2-1).item())})
    means = {mode:{m:float(np.mean([r[m] for r in rs])) for m in ("psnr","ssim","lpips")}
             for mode,rs in rows.items()}
    return means, rows, np.mean(gate_values, 0).tolist()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    paths = sorted(LOW.glob("*.png"))[:128]
    assert all((DDM/p.name).exists() and (RF/p.name).exists() for p in paths)
    train, heldout = paths[:100], paths[100:]
    net = SpatialGate().cuda().train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.RandomState(SEED); history=[]
    for step in range(1, STEPS+1):
        low, ddm, prior, target = make_batch(train, rng)
        output, gates = apply_gate(net, low, ddm, prior)
        pixel = F.l1_loss(output, target)
        grad = (F.l1_loss(output[...,1:,:]-output[...,:-1,:], target[...,1:,:]-target[...,:-1,:])+
                F.l1_loss(output[...,:,1:]-output[...,:,:-1], target[...,:,1:]-target[...,:,:-1]))
        tv = (gates[...,1:,:]-gates[...,:-1,:]).abs().mean() + (gates[...,:,1:]-gates[...,:,:-1]).abs().mean()
        loss = pixel + .1*grad + .005*tv
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 200 == 0:
            row={"step":step,"loss":float(loss),"pixel":float(pixel),"gate_low":float(gates[:,0].mean()),"gate_high":float(gates[:,1].mean())}
            history.append(row); print(json.dumps(row),flush=True)
    import lpips
    net.eval(); perceptual=lpips.LPIPS(net="alex",verbose=False).cuda().eval()
    means, rows, gate_means = evaluate(net, heldout, perceptual)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save({"params":{k:v.cpu() for k,v in net.state_dict().items()},"step":STEPS},OUT/"gate.pt")
    result={"protocol":{"train_images":100,"heldout_images":28,"steps":STEPS,"seed":SEED,
                        "heldout_used_for_training":False,"kernel":31},
            "parameters":sum(p.numel() for p in net.parameters()),"gate_means":gate_means,
            "means":means,"comparisons":{
                "spatial-minus-average":common.metric_comparison(
                    [{**r,"seed":SEED} for r in rows["spatial_gate"]],
                    [{**r,"seed":SEED} for r in rows["average"]]),
                "spatial-minus-ddm":common.metric_comparison(
                    [{**r,"seed":SEED} for r in rows["spatial_gate"]],
                    [{**r,"seed":SEED} for r in rows["ddm"]])},"rows":rows,"history":history}
    common.write_json(OUT/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","history")}),flush=True)


if __name__ == "__main__":
    with open("/tmp/diff-retinex-gpu-0.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
