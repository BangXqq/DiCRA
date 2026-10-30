#!/usr/bin/env python3
"""Zero-training screen of an official RetinexFormer output as a D4-like prior."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import fcntl
import gc
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_response_prior_adapter as common

RF_ROOT = ROOT.parent / "Retinexformer-master"
RF_ARCH = RF_ROOT / "basicsr/models/archs/RetinexFormer_arch.py"
RF_WEIGHT = RF_ROOT / "pretrained_weights/LOL_v1.pth"
INJECTION = os.environ.get("PRIOR_INJECTION", "illumination")
POST_FUSION = INJECTION in ("restoration_residual", "restoration_lowfreq", "frequency_gate", "spatial_gate")
LOWFREQ_FUSION = INJECTION == "restoration_lowfreq"
FREQUENCY_GATE = INJECTION == "frequency_gate"
SPATIAL_GATE = INJECTION == "spatial_gate"
LIGHTWEIGHT_PRIOR = os.environ.get("PRIOR_SOURCE", "retinexformer") == "lightweight"
OUT = ROOT / ("experiments/ddm_spatial_frequency_gate_full_lolv1"
              if SPATIAL_GATE else
              "experiments/ddm_lightweight_frequency_gate_v1"
              if FREQUENCY_GATE and LIGHTWEIGHT_PRIOR else
              "experiments/ddm_retinexformer_frequency_gate_v1"
              if FREQUENCY_GATE else
              "experiments/ddm_retinexformer_restoration_lowfreq_v1"
              if LOWFREQ_FUSION else
              "experiments/ddm_retinexformer_restoration_residual_v1"
              if POST_FUSION else "experiments/ddm_retinexformer_prior_screen_v5")
CACHE = (ROOT / "experiments/lightweight_restoration_prior_v1/cache_lolv1"
         if LIGHTWEIGHT_PRIOR else ROOT / "experiments/retinexformer_lolv1_official_prior/val")
DOSES = (1.0,) if (FREQUENCY_GATE or SPATIAL_GATE) else (0.5,) if POST_FUSION else (0.25, 0.5)
GATE_PATH = ROOT / "experiments/ddm_retinexformer_frequency_gate_train64/gate.json"


def load_retinexformer():
    spec = importlib.util.spec_from_file_location("retinexformer_arch", RF_ARCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.RetinexFormer(
        in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    state = torch.load(RF_WEIGHT, map_location="cpu")
    net.load_state_dict(state["params"], strict=True)
    return net.cuda().eval().requires_grad_(False)


@torch.no_grad()
def retinex_prior(probe, net, low):
    h, w = low.shape[-2:]
    pad_h, pad_w = (-h) % 4, (-w) % 4
    value = F.pad(low, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else low
    enhanced = net(value).clamp(0, 1)[..., :h, :w]
    return enhanced, probe.prior("D2", enhanced, enhanced)


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    historical = json.loads(common.REFERENCE_MANIFEST.read_text())
    cfg = dict(historical["config"])
    cfg["workers"] = 0
    base_config = yaml.safe_load((ROOT / cfg["base_config"]).read_text())
    _, val = common.hist.datasets(base_config)
    source = torch.load(common.SOURCE_CHECKPOINT, map_location="cpu")["params"]
    probe = common.hist.Probe(cfg, base_config, source)
    base_net, state = common.load_denoiser(probe, common.D2_CHECKPOINT)
    assert state["step"] == 10000
    restore_opt = dict(base_config["network_consistentunet"])
    restore_opt.pop("type")
    restore = common.DRNet_Retinex_time(**restore_opt).cuda().eval().requires_grad_(False)
    restore.load_state_dict({k[len("restore_fn."):]: value for k, value in source.items()
                             if k.startswith("restore_fn.")}, strict=True)
    import lpips
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    cached = []
    prior_stats = []
    rf_metrics = []
    with torch.no_grad():
        for index in range(len(val)):
            sample = val[index]
            low = sample["lq_rgb"][None].cuda()
            gt = sample["gt"][None].cuda()
            name = Path(sample["gt_path"]).stem
            cached_image = cv2.imread(str(CACHE / (name + ".png")), cv2.IMREAD_COLOR)
            if cached_image is None:
                raise FileNotFoundError(CACHE / (name + ".png"))
            enhanced = torch.from_numpy(
                cv2.cvtColor(cached_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
            enhanced = enhanced.permute(2, 0, 1)[None].cuda()
            padding = tuple(int(sample[key]) for key in
                            ("pad_left", "pad_right", "pad_top", "pad_bottom"))
            if any(padding):
                enhanced = F.pad(enhanced, padding, mode="reflect")
            if enhanced.shape[-2:] != low.shape[-2:]:
                raise ValueError((enhanced.shape, low.shape, padding))
            rf_prior = probe.prior("D2", enhanced, enhanced)
            d2, d4 = probe.prior("D2", low, low), probe.prior("D4", low, gt)
            cached.append({"sample": sample, "name": name,
                           "condition01": sample["lq"][None].cpu(), "d2": d2.cpu(),
                           "residual": (rf_prior - d2).cpu(),
                           "enhanced": enhanced.cpu()})
            prior_stats.append({
                "image": cached[-1]["name"],
                "rf_D4_mae": float((rf_prior - d4).abs().mean().item()),
                "D2_D4_mae": float((d2 - d4).abs().mean().item()),
            })
            gt_eval = common.hist.unpad(gt, sample)
            enhanced_eval = common.hist.unpad(enhanced, sample)
            image, target = common.hist.uint_image(enhanced_eval), common.hist.uint_image(gt_eval)
            rf_metrics.append({
                "image": cached[-1]["name"],
                "psnr": float(calculate_psnr(image, target, crop_border=0)),
                "ssim": float(calculate_ssim(image, target, crop_border=0)),
                "lpips": float(perceptual(enhanced_eval * 2 - 1, gt_eval * 2 - 1).item()),
            })

    del enhanced, enhanced_eval, gt, gt_eval, low, rf_prior, d2, d4
    gc.collect()
    torch.cuda.empty_cache()

    results = {}
    gate = json.loads(GATE_PATH.read_text()) if FREQUENCY_GATE else None
    spatial_gate = None
    if SPATIAL_GATE:
        from tools.train_spatial_frequency_gate import SpatialGate
        spatial_gate = SpatialGate().cuda().eval().requires_grad_(False)
        spatial_gate.load_state_dict(torch.load(
            ROOT / "experiments/spatial_frequency_gate_pareto_full_v1/gate.pt",
            map_location="cpu")["params"], strict=True)
    for dose in DOSES:
        rows = {"bypass": [], "matched": [], "shuffled": []}
        for position, item in enumerate(cached):
            sample = item["sample"]
            gt = sample["gt"][None].cuda()
            condition01 = item["condition01"].cuda()
            current_d2 = item["d2"].cuda()
            cond = condition01 * 2 - 1
            gt_eval = common.hist.unpad(gt, sample)
            sources = {"bypass": position, "matched": position,
                       "shuffled": (position + 1) % len(cached)}
            for mode, source_position in sources.items():
                source_item = cached[source_position]
                assert source_item["residual"].shape == item["d2"].shape
                prior = current_d2 if (mode == "bypass" or POST_FUSION) else (
                    current_d2 + dose * source_item["residual"].cuda()).clamp(0, 1)
                for seed in cfg["eval_seeds"]:
                    bank = common.hist.noise_bank(OUT, item["name"], seed, gt.shape)
                    x = bank["initial"].cuda()
                    for j, t in enumerate(common.TIMESTEPS):
                        a = probe.alpha[t]
                        ap = (probe.alpha[common.TIMESTEPS[j + 1]] if j < 4
                              else torch.ones_like(a))
                        gamma = a.sqrt().reshape(1, 1)
                        used_prior = prior if t in common.ACTIVE_TIMESTEPS else current_d2
                        pred = probe.predict(base_net, cond, x, gamma, used_prior)
                        x0 = ((x - (1 - a).sqrt() * pred) / a.sqrt()).clamp(-1, 1)
                        fixed = restore(condition01, (x0 + 1) / 2, gamma)[0]
                        fixed = fixed.clamp(0, 1) * 2 - 1
                        sigma = ((1 - ap) / (1 - a) * (1 - a / ap)).sqrt()
                        x = (ap.sqrt() * fixed + (1 - ap - sigma.square()).sqrt() * pred
                             + sigma * bank["steps"][j].cuda())
                    restored = common.hist.unpad((x + 1) / 2, sample).clamp(0, 1)
                    if POST_FUSION and mode != "bypass":
                        source_enhanced = common.hist.unpad(
                            source_item["enhanced"].cuda(), source_item["sample"])
                        if source_enhanced.shape != restored.shape:
                            raise ValueError((source_enhanced.shape, restored.shape))
                        if SPATIAL_GATE:
                            from tools.train_spatial_frequency_gate import apply_gate
                            current_low = common.hist.unpad(condition01[:, :3], sample)
                            restored, _ = apply_gate(
                                spatial_gate, current_low, restored, source_enhanced)
                        elif FREQUENCY_GATE:
                            radius = 15
                            residual = source_enhanced - restored
                            low_residual = F.avg_pool2d(
                                F.pad(residual, (radius,) * 4, mode="reflect"),
                                31, stride=1)
                            high_residual = residual - low_residual
                            restored = (restored + gate["low_weight"] * low_residual
                                        + gate["high_weight"] * high_residual).clamp(0, 1)
                        elif LOWFREQ_FUSION:
                            # Inject only low-frequency RGB statistics and keep
                            # the DDM high-frequency structure unchanged.
                            radius = 15
                            low_source = F.avg_pool2d(
                                F.pad(source_enhanced, (radius,) * 4, mode="reflect"),
                                31, stride=1)
                            low_current = F.avg_pool2d(
                                F.pad(restored, (radius,) * 4, mode="reflect"),
                                31, stride=1)
                            restored = (restored + dose * (low_source - low_current)).clamp(0, 1)
                        else:
                            # A parameter-free residual adapter in restoration
                            # space: y = y_DDM + dose * (P_RF - y_DDM).
                            restored = (restored + dose * (source_enhanced - restored)).clamp(0, 1)
                    image, target = common.hist.uint_image(restored), common.hist.uint_image(gt_eval)
                    rows[mode].append({
                        "image": item["name"], "seed": seed,
                        "psnr": float(calculate_psnr(image, target, crop_border=0)),
                        "ssim": float(calculate_ssim(image, target, crop_border=0)),
                        "lpips": float(perceptual(restored * 2 - 1, gt_eval * 2 - 1).item()),
                        "noise_sha256": bank["sha256"],
                    })
            print(json.dumps({"event": "eval", "dose": dose,
                              "completed_images": position + 1}), flush=True)
        results[str(dose)] = {
            "means": {mode: {metric: float(np.mean([r[metric] for r in values]))
                              for metric in ("psnr", "ssim", "lpips")}
                      for mode, values in rows.items()},
            "comparisons": {
                "matched-minus-bypass": common.metric_comparison(rows["matched"], rows["bypass"]),
                "matched-minus-shuffled": common.metric_comparison(rows["matched"], rows["shuffled"]),
            }, "rows": rows,
        }

    reference = json.loads(common.D2_EVALUATION.read_text())["system"]
    max_error = max(abs(a["psnr"] - b["psnr"])
                    for a, b in zip(results[str(DOSES[0])]["rows"]["bypass"], reference))
    assert max_error < 1e-4
    summary = {
        "retinexformer_output_means": {
            metric: float(np.mean([r[metric] for r in rf_metrics]))
            for metric in ("psnr", "ssim", "lpips")},
        "prior_means": {
            key: float(np.mean([r[key] for r in prior_stats]))
            for key in ("rf_D4_mae", "D2_D4_mae")},
        "results": results,
        "manifest": {
            "retinexformer_arch": str(RF_ARCH), "retinexformer_weight": str(RF_WEIGHT),
            "arch_sha256": common.digest(RF_ARCH), "weight_sha256": common.digest(RF_WEIGHT),
            "cache": str(CACHE.relative_to(ROOT)),
            "injection": INJECTION,
            "frequency_gate": gate,
            "cache_sha256": {path.name: common.digest(path) for path in sorted(CACHE.glob("*.png"))},
            "D2_checkpoint_sha256": common.digest(common.D2_CHECKPOINT),
            "training": "none", "active_timesteps": sorted(common.ACTIVE_TIMESTEPS, reverse=True),
        },
    }
    common.write_json(OUT / "summary.json", summary)
    print(json.dumps({"event": "complete", "summary": {
        "retinexformer_output_means": summary["retinexformer_output_means"],
        "prior_means": summary["prior_means"],
        "results": {key: value["means"] for key, value in results.items()}}}), flush=True)


if __name__ == "__main__":
    with open("/tmp/diff-retinex-gpu-0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
