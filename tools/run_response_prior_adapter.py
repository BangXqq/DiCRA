#!/usr/bin/env python3
"""Deployable low-light prior injection by oracle-response distillation.

The historical D2 denoiser is frozen.  A zero-initialized, multi-scale prior
adapter reads only low-light-derived tensors and adds a residual to D2's noise
prediction at the three high-noise DDIM steps.  The adapter is supervised by
both the true diffusion noise and the matched oracle-high3 denoiser response.

This experiment deliberately tests one branch only.  It supports a resumable
2K screen before the locked 10K run and evaluates matched, cyclically shuffled,
and exact-bypass controls with the historical noise banks.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
HIST_ROOT = ROOT / "experiments/baseline_rollback_20260916_214256/source_before_rollback"
sys.path.insert(0, str(ROOT))

# Load the immutable historical experiment implementation without allowing the
# current data package to shadow its dataset implementation.
legacy_data = types.ModuleType("data")
legacy_data.__path__ = [str(HIST_ROOT / "data")]
sys.modules["data"] = legacy_data
spec = importlib.util.spec_from_file_location(
    "historical_d4_probe", HIST_ROOT / "scripts/ddm_prior_probe_gtgray.py")
hist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hist)
hist.ROOT = ROOT

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml
from basicsr.metrics import calculate_psnr, calculate_ssim
from archs.DRNet_Retinex_arch import DRNet_Retinex_time


OUT = ROOT / "experiments/ddm_response_prior_adapter_v4"
SMOKE_OUT = ROOT / "experiments/ddm_response_prior_adapter_v4_smoke"
SOURCE_CHECKPOINT = ROOT / "experiments/diff-retinex_plus_lolv1_resume395k/models/net_g_ddpm_170000.pth"
D2_CHECKPOINT = ROOT / "experiments/ddm_prior_probe_v1/D2/step_010000.pt"
TEACHER_CHECKPOINT = ROOT / "experiments/ddm_d4_gated_v1/high3/latest.pt"
REFERENCE_MANIFEST = ROOT / "experiments/ddm_prior_probe_gtgray_v1/manifest.json"
D2_EVALUATION = ROOT / "experiments/ddm_prior_probe_v1/D2/eval_010000.json"
TIMESTEPS = (1999, 1599, 1199, 799, 399)
ACTIVE_TIMESTEPS = {1999, 1599, 1199}
TRAIN_THRESHOLD = 1000
RESPONSE_WEIGHT = 1.0
ADAPTER_LR = 3e-5


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


class TimeAffine(nn.Module):
    def __init__(self, channels, time_channels=64):
        super().__init__()
        self.project = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, channels * 2))

    def forward(self, x, embedding):
        scale, shift = self.project(embedding).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels=64):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.affine = TimeAffine(out_channels, time_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.affine(h, embedding)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class LowLightPriorAdapter(nn.Module):
    """Multi-scale low-only prior encoder with a zero residual output head."""
    def __init__(self, in_channels=7, base=32, time_channels=64):
        super().__init__()
        self.time = nn.Sequential(
            nn.Linear(1, time_channels), nn.SiLU(),
            nn.Linear(time_channels, time_channels))
        self.stem = nn.Conv2d(in_channels, base, 3, padding=1)
        self.enc0 = ResidualBlock(base, base, time_channels)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc1 = ResidualBlock(base * 2, base * 2, time_channels)
        self.down2 = nn.Conv2d(base * 2, base * 3, 3, stride=2, padding=1)
        self.mid = ResidualBlock(base * 3, base * 3, time_channels)
        self.up1 = nn.Conv2d(base * 3, base * 2, 3, padding=1)
        self.dec1 = ResidualBlock(base * 4, base * 2, time_channels)
        self.up0 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec0 = ResidualBlock(base * 2, base, time_channels)
        self.head = nn.Conv2d(base, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, condition01, d2, gamma):
        # condition01 is the deployable six-channel Retinex condition already
        # produced from the low-light input; D2 is its smoothed RGB luminance.
        embedding = self.time(gamma.reshape(len(gamma), 1))
        e0 = self.enc0(self.stem(torch.cat([condition01, d2], dim=1)), embedding)
        e1 = self.enc1(self.down1(e0), embedding)
        mid = self.mid(self.down2(e1), embedding)
        # Nearest-neighbour upsampling keeps the audited deterministic CUDA mode
        # supported on the project's pinned PyTorch 2.0 environment.
        u1 = F.interpolate(mid, size=e1.shape[-2:], mode="nearest")
        u1 = self.dec1(torch.cat([self.up1(u1), e1], dim=1), embedding)
        u0 = F.interpolate(u1, size=e0.shape[-2:], mode="nearest")
        u0 = self.dec0(torch.cat([self.up0(u0), e0], dim=1), embedding)
        return self.head(F.silu(u0))


def training_noise(probe, step, gt):
    rng = np.random.RandomState(hist.seed_for(probe.cfg["seed"], "noise_level", step))
    t = int(rng.randint(1, 2001))
    gamma = torch.tensor(
        rng.uniform(probe.gamma_prev[t - 1], probe.gamma_prev[t], size=(len(gt), 1)),
        dtype=torch.float32, device=probe.device)
    generator = torch.Generator(device="cpu").manual_seed(
        hist.seed_for(probe.cfg["seed"], "epsilon", step))
    eps = torch.randn(gt.shape, generator=generator).to(probe.device)
    g = gamma[:, :, None, None]
    xt = g * (gt * 2 - 1) + (1 - g.square()).sqrt() * eps
    return t, xt, gamma, eps


def load_denoiser(probe, checkpoint):
    net = probe.net("D4")
    state = torch.load(checkpoint, map_location="cpu")
    net.load_state_dict(state["params"], strict=True)
    return net.eval().requires_grad_(False), state


def save_checkpoint(path, adapter, optimizer, step, protocol):
    hist.atomic_save({
        "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(), "step": step,
        "protocol_sha256": protocol, "diagnostic_gt_condition": False,
        "base_frozen": True, "teacher_used_only_during_training": True,
    }, path)


def metric_comparison(rows_a, rows_b):
    keys_a = [(r["image"], r["seed"]) for r in rows_a]
    keys_b = [(r["image"], r["seed"]) for r in rows_b]
    assert keys_a == keys_b
    result = {}
    for metric in ("psnr", "ssim", "lpips"):
        per_image = {}
        for a, b in zip(rows_a, rows_b):
            per_image.setdefault(a["image"], []).append(a[metric] - b[metric])
        delta = np.array([np.mean(values) for values in per_image.values()])
        rng = np.random.RandomState(2026)
        boot = delta[rng.randint(0, len(delta), (20000, len(delta)))].mean(1)
        result[metric] = {
            "mean": float(delta.mean()),
            "image_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
            "improved_images": int((delta < 0).sum() if metric == "lpips" else (delta > 0).sum()),
        }
    return result


@torch.no_grad()
def evaluate(probe, base_net, adapter, restore, val, out, target, perceptual, smoke):
    adapter.eval()
    indices = list(range(1) if smoke else range(len(val)))
    seeds = probe.cfg["eval_seeds"][:1] if smoke else probe.cfg["eval_seeds"]
    samples = []
    for index in indices:
        sample = val[index]
        samples.append({
            "sample": sample,
            "name": Path(sample["gt_path"]).stem,
            "condition01": sample["lq"][None].cuda(),
            "low": sample["lq_rgb"][None].cuda(),
        })
    for item in samples:
        item["d2"] = probe.prior("D2", item["low"], item["low"])

    rows = {"bypass": [], "matched": [], "shuffled": []}
    began = time.monotonic()
    for position, item in enumerate(samples):
        sample = item["sample"]
        gt = sample["gt"][None].cuda()
        cond = item["condition01"] * 2 - 1
        gt_eval = hist.unpad(gt, sample)
        source_positions = {
            "bypass": position,
            "matched": position,
            "shuffled": (position + 1) % len(samples) if len(samples) > 1 else position,
        }
        for mode, source_position in source_positions.items():
            source = samples[source_position]
            assert source["condition01"].shape[-2:] == item["condition01"].shape[-2:]
            for seed in seeds:
                bank = hist.noise_bank(out, item["name"], seed, gt.shape)
                x = bank["initial"].cuda()
                for j, t in enumerate(TIMESTEPS):
                    a = probe.alpha[t]
                    ap = probe.alpha[TIMESTEPS[j + 1]] if j < 4 else torch.ones_like(a)
                    gamma = a.sqrt().reshape(1, 1)
                    base_pred = probe.predict(base_net, cond, x, gamma, item["d2"])
                    if mode != "bypass" and t in ACTIVE_TIMESTEPS:
                        delta = adapter(source["condition01"], source["d2"], gamma)
                        pred = base_pred + delta
                    else:
                        pred = base_pred
                    x0 = ((x - (1 - a).sqrt() * pred) / a.sqrt()).clamp(-1, 1)
                    fixed, _, _, _ = restore(item["condition01"], (x0 + 1) / 2, gamma)
                    fixed = fixed.clamp(0, 1) * 2 - 1
                    sigma = ((1 - ap) / (1 - a) * (1 - a / ap)).sqrt()
                    x = (ap.sqrt() * fixed + (1 - ap - sigma.square()).sqrt() * pred
                         + sigma * bank["steps"][j].cuda())
                result = hist.unpad((x + 1) / 2, sample).clamp(0, 1)
                image, target_image = hist.uint_image(result), hist.uint_image(gt_eval)
                rows[mode].append({
                    "image": item["name"], "seed": seed,
                    "psnr": float(calculate_psnr(image, target_image, crop_border=0)),
                    "ssim": float(calculate_ssim(image, target_image, crop_border=0)),
                    "lpips": perceptual(result * 2 - 1, gt_eval * 2 - 1).item(),
                    "noise_sha256": bank["sha256"],
                    "prior_source": source["name"] if mode != "bypass" else None,
                })
        print(json.dumps({"event": "eval", "step": target, "completed_images": position + 1,
                          "elapsed_seconds": round(time.monotonic() - began, 1)}), flush=True)

    means = {mode: {metric: float(np.mean([r[metric] for r in values]))
                    for metric in ("psnr", "ssim", "lpips")}
             for mode, values in rows.items()}
    result = {
        "step": target, "rows": rows, "means": means,
        "comparisons": {
            "matched-minus-bypass": metric_comparison(rows["matched"], rows["bypass"]),
            "matched-minus-shuffled": metric_comparison(rows["matched"], rows["shuffled"]),
        },
    }
    if not smoke:
        reference = json.loads(D2_EVALUATION.read_text())
        reference_rows = reference["system"]
        assert [(r["image"], r["seed"]) for r in rows["bypass"]] == [
            (r["image"], r["seed"]) for r in reference_rows]
        result["bypass-minus-historical-D2"] = metric_comparison(rows["bypass"], reference_rows)
        max_abs = {metric: max(abs(a[metric] - b[metric])
                               for a, b in zip(rows["bypass"], reference_rows))
                   for metric in ("psnr", "ssim", "lpips")}
        result["historical_D2_max_abs_metric_error"] = max_abs
        assert max_abs["psnr"] < 1e-4 and max_abs["ssim"] < 1e-6 and max_abs["lpips"] < 1e-6
    write_json(out / ("evaluation_%06d.json" % target), result)
    adapter.train()
    return result


def train(probe, train_data, val, out, protocol, target, smoke, restore, perceptual):
    base_net, base_state = load_denoiser(probe, D2_CHECKPOINT)
    teacher_net, teacher_state = load_denoiser(probe, TEACHER_CHECKPOINT)
    assert base_state["step"] == teacher_state["step"] == 10000
    adapter = LowLightPriorAdapter().cuda()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPTER_LR, betas=(0.9, 0.99))
    latest = out / "latest.pt"
    start = 0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu")
        assert saved["protocol_sha256"] == protocol
        adapter.load_state_dict(saved["adapter"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start = int(saved["step"])
    if start > target:
        raise RuntimeError("Checkpoint step exceeds requested target")

    # Zero-head initialization or a saved step-0 adapter must exactly preserve D2.
    sample = train_data[0]
    with torch.no_grad():
        condition01 = sample["lq"][None].cuda()
        low = sample["lq_rgb"][None].cuda()
        d2 = probe.prior("D2", low, low)
        zero_error = adapter(condition01, d2, torch.ones((1, 1), device="cuda")).abs().max().item()
    if start == 0:
        assert zero_error == 0.0

    reference_logs = [json.loads(line) for line in
                      (ROOT / "experiments/ddm_prior_probe_v1/D0/train.jsonl").read_text().splitlines()]
    reference_hashes = {row["step"]: row["input_sha256"] for row in reference_logs}
    dataset = hist.PairedSamples(
        train_data, start * probe.cfg["batch_size"], target * probe.cfg["batch_size"], probe.cfg["seed"])
    loader = DataLoader(dataset, batch_size=probe.cfg["batch_size"], shuffle=False,
                        num_workers=0 if smoke else probe.cfg["workers"], pin_memory=True,
                        generator=torch.Generator().manual_seed(probe.cfg["seed"]))
    log_path = out / "train.jsonl"
    began = time.monotonic()
    active_updates = 0
    adapter.train()
    for step, batch in enumerate(loader, start=start + 1):
        gt = batch["gt"].cuda()
        condition01 = batch["lq"].cuda()
        cond = condition01 * 2 - 1
        low = batch["lq_rgb"].cuda()
        t, xt, gamma, eps = training_noise(probe, step, gt)
        d2 = probe.prior("D2", low, low)
        input_hash = hist.tensor_digest(cond, low, gt, gamma, eps)
        if step in reference_hashes:
            assert input_hash == reference_hashes[step], ("training stream mismatch", step)

        row = {"event": "train", "step": step, "target": target, "t": t,
               "active": bool(t >= TRAIN_THRESHOLD), "input_sha256": input_hash}
        if t >= TRAIN_THRESHOLD:
            d4 = probe.prior("D4", low, gt)
            with torch.no_grad():
                base_pred = probe.predict(base_net, cond, xt, gamma, d2)
                teacher_pred = probe.predict(teacher_net, cond, xt, gamma, d4)
            delta = adapter(condition01, d2, gamma)
            student_pred = base_pred + delta
            noise_loss = F.l1_loss(student_pred, eps)
            # The oracle response is only about 1e-3. Smooth-L1 would enter its
            # quadratic region and make this signal orders of magnitude weaker
            # than the ordinary noise L1 objective.
            response_loss = F.l1_loss(delta, teacher_pred - base_pred)
            loss = noise_loss + RESPONSE_WEIGHT * response_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
            assert math.isfinite(grad_norm)
            optimizer.step()
            active_updates += 1
            row.update({
                "loss": float(loss.item()), "noise_loss": float(noise_loss.item()),
                "response_loss": float(response_loss.item()), "grad_norm": grad_norm,
                "delta_abs_mean": float(delta.detach().abs().mean().item()),
                "teacher_delta_abs_mean": float((teacher_pred - base_pred).abs().mean().item()),
            })
        if step == start + 1 or step % (1 if smoke else 100) == 0:
            row["active_updates_this_run"] = active_updates
            with open(log_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            write_json(out / "status.json", row)
            print(json.dumps(row), flush=True)
        if step % (10 if smoke else 500) == 0 or step == target:
            save_checkpoint(latest, adapter, optimizer, step, protocol)

    saved = torch.load(latest, map_location="cpu")
    assert saved["step"] == target
    result = evaluate(probe, base_net, adapter, restore, val, out, target, perceptual, smoke)
    write_json(out / "summary.json", {
        "step": target, "means": result["means"], "comparisons": result["comparisons"],
        "screen_pass": bool(
            result["comparisons"]["matched-minus-bypass"]["psnr"]["mean"] > 0 and
            result["comparisons"]["matched-minus-shuffled"]["psnr"]["mean"] > 0),
        "success_definition": "matched PSNR > D2 bypass and matched PSNR > shuffled",
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--target", type=int, default=2000, choices=(10, 2000, 10000))
    args = parser.parse_args()
    if args.smoke:
        args.target = 10
    out = SMOKE_OUT if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    historical = json.loads(REFERENCE_MANIFEST.read_text())
    cfg = dict(historical["config"])
    cfg["workers"] = 0 if args.smoke else cfg["workers"]
    base_config = yaml.safe_load((ROOT / cfg["base_config"]).read_text())
    train_data, val = hist.datasets(base_config)
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu")["params"]
    probe = hist.Probe(cfg, base_config, source)

    restore_opt = dict(base_config["network_consistentunet"])
    restore_opt.pop("type")
    restore = DRNet_Retinex_time(**restore_opt).cuda().eval().requires_grad_(False)
    restore.load_state_dict({k[len("restore_fn."):]: value for k, value in source.items()
                             if k.startswith("restore_fn.")}, strict=True)
    import lpips
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    paths = [Path(__file__).resolve(),
             HIST_ROOT / "scripts/ddm_prior_probe_gtgray.py",
             HIST_ROOT / "data/lol_dataset.py", ROOT / "archs/sr3unet_arch.py",
             ROOT / "archs/DRNet_Retinex_arch.py", ROOT / "archs/MoE_arch.py",
             ROOT / "scripts/utils.py", ROOT / cfg["base_config"], SOURCE_CHECKPOINT,
             D2_CHECKPOINT, TEACHER_CHECKPOINT, REFERENCE_MANIFEST, D2_EVALUATION]
    manifest = {
        "experiment": "frozen D2 plus deployable low-light multi-scale response-prior adapter",
        "smoke": args.smoke, "max_target": 10000, "screen_target": 2000,
        "active_timesteps": sorted(ACTIVE_TIMESTEPS, reverse=True),
        "train_threshold": TRAIN_THRESHOLD, "response_weight": RESPONSE_WEIGHT,
        "adapter_lr": ADAPTER_LR,
        "adapter_input": "six-channel low-light condition plus low-light D2 only",
        "base_frozen": True, "teacher": "oracle high3; training supervision only",
        "training_stream": "paired with historical D0/D2 seed 1234",
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    protocol = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        write_json(manifest_path, manifest)
    write_json(out / "qa.json", {
        "D2_checkpoint_step": 10000, "teacher_checkpoint_step": 10000,
        "adapter_zero_initialized": True, "GT_unavailable_to_adapter": True,
        "bypass_must_match_historical_D2": True,
    })
    result = train(probe, train_data, val, out, protocol, args.target, args.smoke, restore, perceptual)
    for path, expected in manifest["sources"].items():
        assert digest(ROOT / path) == expected
    print(json.dumps({"event": "complete", "step": args.target,
                      "means": result["means"], "comparisons": result["comparisons"]}), flush=True)


if __name__ == "__main__":
    with open("/tmp/diff-retinex-gpu-0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
