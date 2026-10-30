#!/usr/bin/env python3
"""Pre-registered pixel/SSIM/LPIPS spatial-gate feasibility experiment."""
import fcntl
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_response_prior_adapter as common
from tools import train_spatial_frequency_gate as base

OUT = ROOT / "experiments/spatial_frequency_gate_pareto_v1"
STEPS, BATCH, CROP, SEED = 2000, 4, 128, 20260921


def ssim_loss(x, y):
    mu_x, mu_y = F.avg_pool2d(x, 11, 1, 5), F.avg_pool2d(y, 11, 1, 5)
    var_x = F.avg_pool2d(x*x, 11, 1, 5) - mu_x.square()
    var_y = F.avg_pool2d(y*y, 11, 1, 5) - mu_y.square()
    cov = F.avg_pool2d(x*y, 11, 1, 5) - mu_x*mu_y
    score = ((2*mu_x*mu_y + .01**2)*(2*cov + .03**2) /
             ((mu_x.square()+mu_y.square()+.01**2)*(var_x+var_y+.03**2)))
    return 1-score.mean()


def make_batch(paths, rng):
    values=[[],[],[],[]]
    for index in rng.randint(0,len(paths),BATCH):
        sample=base.tensors(paths[int(index)])
        top,left=rng.randint(0,273),rng.randint(0,473)
        sample=[x[:,top:top+CROP,left:left+CROP] for x in sample]
        if rng.rand()<.5: sample=[x.flip(-1) for x in sample]
        for bucket,value in zip(values,sample): bucket.append(value)
    return [torch.stack(x).cuda() for x in values]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    paths=sorted(base.LOW.glob("*.png"))[128:256]
    assert len(paths)==128 and all((base.DDM/p.name).exists() for p in paths)
    train,heldout=paths[:100],paths[100:]
    net=base.SpatialGate().cuda().train()
    # Start exactly from the simple average rather than the earlier global gate.
    net.body[-1].weight.data.zero_(); net.body[-1].bias.data.zero_()
    import lpips
    perceptual=lpips.LPIPS(net="alex",verbose=False).cuda().eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.RandomState(SEED); history=[]
    for step in range(1,STEPS+1):
        low,ddm,prior,target=make_batch(train,rng)
        output,gates=base.apply_gate(net,low,ddm,prior)
        pixel=F.l1_loss(output,target)
        structure=ssim_loss(output,target)
        perception=perceptual(output*2-1,target*2-1).mean()
        tv=(gates[...,1:,:]-gates[...,:-1,:]).abs().mean()+(gates[...,:,1:]-gates[...,:,:-1]).abs().mean()
        # Fixed before evaluation; no coefficient sweep.
        loss=pixel+.2*structure+.05*perception+.002*tv
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step%100==0:
            row={"step":step,"loss":float(loss),"pixel":float(pixel),"ssim_loss":float(structure),
                 "lpips":float(perception),"gate_low":float(gates[:,0].mean()),"gate_high":float(gates[:,1].mean())}
            history.append(row); print(json.dumps(row),flush=True)
    net.eval(); means,rows,gate_means=base.evaluate(net,heldout,perceptual)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save({"params":{k:v.cpu() for k,v in net.state_dict().items()},"step":STEPS},OUT/"gate.pt")
    tagged=lambda rs:[{**r,"seed":SEED} for r in rs]
    result={"protocol":{"train_images":100,"heldout_images":28,"source_indices":[128,256],
                        "steps":STEPS,"seed":SEED,"loss":"L1 + 0.2*DSSIM + 0.05*LPIPS + 0.002*gateTV",
                        "heldout_used_for_training":False},"parameters":sum(p.numel() for p in net.parameters()),
            "gate_means":gate_means,"means":means,"comparisons":{
                "spatial-minus-average":common.metric_comparison(tagged(rows["spatial_gate"]),tagged(rows["average"])),
                "spatial-minus-ddm":common.metric_comparison(tagged(rows["spatial_gate"]),tagged(rows["ddm"]))},
            "rows":rows,"history":history}
    common.write_json(OUT/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","history")}),flush=True)


if __name__=="__main__":
    with open("/tmp/diff-retinex-gpu-0.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
