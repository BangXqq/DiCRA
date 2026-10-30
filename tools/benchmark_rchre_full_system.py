#!/usr/bin/env python3
"""Unified full-system benchmark for the final Diff-Retinex+++RCHRE method."""
import importlib.util
import json
import logging
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RF_ROOT = ROOT.parent / "Retinexformer-master"
OUT = ROOT / "reproduction_records/rchre_full_system_complexity_20260924"


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def timed(fn, warmup, repeats):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(repeats): fn()
    end.record(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "latency_ms": float(start.elapsed_time(end) / repeats),
        "peak_allocated_mib": float(peak / 2**20),
        "incremental_peak_mib": float((peak - base) / 2**20),
    }


def profile_macs(fn):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        with_flops=True) as profiler:
        fn(); torch.cuda.synchronize()
    flops = sum(event.flops for event in profiler.key_averages())
    return {"profiler_flops": int(flops), "profiler_gmac": float(flops / 2e9)}


@torch.no_grad()
def main():
    import archs, models, data  # register project classes
    from basicsr.models import build_model
    from basicsr.utils import get_root_logger, set_random_seed
    from tools.train_rchre_validation import RCHRE

    config = yaml.safe_load((ROOT / "options/test_diff_retinex_plus_lolv1.yml").read_text())
    config.update(is_train=False, dist=False, rank=0, world_size=1)
    config["val"]["metrics"] = None
    config["val"]["save_img"] = False
    get_root_logger().setLevel(logging.WARNING)
    set_random_seed(0)
    baseline = build_model(config)

    spec = importlib.util.spec_from_file_location(
        "retinexformer_arch", RF_ROOT / "basicsr/models/archs/RetinexFormer_arch.py")
    rf_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf_module)
    prior = rf_module.RetinexFormer(
        in_channels=3, out_channels=3, n_feat=40, stage=1,
        num_blocks=[1, 2, 2]).cuda().eval().requires_grad_(False)
    prior.load_state_dict(torch.load(
        RF_ROOT / "pretrained_weights/LOL_v1.pth", map_location="cpu")["params"])
    fusion = RCHRE().cuda().eval().requires_grad_(False)
    fusion.load_state_dict(torch.load(
        ROOT / "experiments/rchre_validation_v1/model.pt", map_location="cpu")["params"], strict=True)

    low = torch.rand(1, 3, 400, 600, device="cuda")
    # The official test loader concatenates RGB and its histogram-equalized
    # RGB view before the 9-channel conditional diffusion input is formed.
    baseline_lq = torch.cat([low, low], 1)
    baseline.feed_data({"lq": baseline_lq, "gt": low})
    prior_cache = prior(low)
    anchor_cache = low.clone()

    def run_baseline():
        baseline.feed_data({"lq": baseline_lq, "gt": low})
        baseline.test()
        return baseline.diff_sr

    def run_prior(): return prior(low)
    def run_fusion(): return fusion(low, anchor_cache, prior_cache)[0]
    def run_full():
        anchor = run_baseline()
        candidate = prior(low)
        return fusion(low, anchor, candidate)[0]

    functions = {
        "diff_retinex_plus": (run_baseline, 3, 10),
        "retinexformer": (run_prior, 10, 50),
        "rchre": (run_fusion, 20, 100),
        "full_system": (run_full, 3, 10),
    }
    records = {}
    for name, (fn, warmup, repeats) in functions.items():
        record = timed(fn, warmup, repeats)
        record.update(profile_macs(fn))
        records[name] = record
        print(name, json.dumps(record), flush=True)

    parameters = {
        "diff_retinex_plus": count_params(baseline.bare_ddpm_model),
        "retinexformer": count_params(prior),
        "rchre": count_params(fusion),
    }
    parameters["full_system"] = sum(parameters.values())
    report = {
        "protocol": {
            "input_shape": [1, 3, 400, 600],
            "device": torch.cuda.get_device_name(0),
            "dtype": "FP32",
            "batch_size": 1,
            "diffusion": {"ddim_timesteps": 5, "ddim_eta": 1.0},
            "timing": "CUDA events; synchronized; excludes image I/O and metric computation",
            "macs": "PyTorch profiler-reported FLOPs divided by two; unsupported operators may be omitted",
            "memory": "incremental peak allocated CUDA memory above loaded models and resident inputs",
        },
        "parameters": parameters,
        "records": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
