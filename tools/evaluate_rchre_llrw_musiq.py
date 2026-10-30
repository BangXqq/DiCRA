#!/usr/bin/env python3
"""Frozen LOL-v1 models on unpaired LLRW; native-size PNG MUSIQ evaluation."""
import argparse
import hashlib
import importlib
import json
import logging
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reproduction_records/rchre_llrw_musiq_20260923'
IMAGES = ROOT / 'results/RCHRE_LLRW_lolv1_seed0'
DATA = ROOT / 'dataset/LLRW'
DDPM = ROOT / 'pretrained_weights/lolv1/net_g_ddpm_val_lolv1.pth'
RCHRE_WEIGHT = ROOT / 'experiments/rchre_validation_v1/model.pt'
RF_ROOT = ROOT.parent / 'Retinexformer-master'
RF_WEIGHT = RF_ROOT / 'pretrained_weights/LOL_v1.pth'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1)[None].cuda().float() / 255


def save(value, path):
    rgb = value.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    bgr = cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(path), bgr)


@torch.no_grad()
def enhance():
    import archs, models, data
    from basicsr.models import build_model
    from basicsr.utils import set_random_seed, get_root_logger
    from data.lol_dataset import LoLDataset
    from torch.utils.data import DataLoader
    from tools.train_rchre_validation import RCHRE
    from tools.screen_retinexformer_ddm_prior import load_retinexformer
    cfg = yaml.safe_load((ROOT / 'options/test_diff-retinex_LLRW.yml').read_text())
    cfg.update(is_train=False, dist=False, rank=0, world_size=1)
    cfg['path']['pretrain_network_ddpm'] = str(DDPM)
    cfg['val']['metrics'] = None
    get_root_logger().setLevel(logging.WARNING)
    set_random_seed(0)
    model = build_model(cfg)
    dsopt = cfg['datasets']['val']
    dsopt.update(gt_root=str(DATA), input_root=str(DATA), phase='test')
    dataset = LoLDataset(dsopt)
    dataset.gt_paths = sorted(dataset.gt_paths)
    rf = load_retinexformer()
    module = RCHRE().cuda().eval().requires_grad_(False)
    module.load_state_dict(torch.load(RCHRE_WEIGHT, map_location='cpu')['params'], strict=True)
    for method in ('baseline', 'prior', 'rchre'):
        (IMAGES / method).mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        'dataset': str(DATA), 'n_images': len(dataset),
        'protocol': 'Native 640x480; frozen LOL-v1 checkpoints; no LLRW training or tuning',
        'diffusion': {'ddim_timesteps': 5, 'ddim_eta': 1.0, 'seed_per_image': 0},
        'gt': 'None. Input is dataset loader placeholder only; no reference metrics or GT adjustment.',
        'weights': {str(p): digest(p) for p in (DDPM, RF_WEIGHT, RCHRE_WEIGHT)},
        'inputs': {Path(p).name: digest(p) for p in dataset.gt_paths},
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for index, sample in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
        path = Path(sample['lq_path'][0])
        set_random_seed(0)
        model.feed_data(sample)
        model.test()
        save(model.diff_sr, IMAGES / 'baseline' / path.name)
        low = tensor(path)
        # Quantize the prior and anchor exactly as in previous cached evaluations.
        save(rf(low), IMAGES / 'prior' / path.name)
        baseline, prior = tensor(IMAGES / 'baseline' / path.name), tensor(IMAGES / 'prior' / path.name)
        output, _, _, _ = module(low, baseline, prior)
        save(output, IMAGES / 'rchre' / path.name)
        print(f'enhance {index + 1}/{len(dataset)} {path.name}', flush=True)


def load_musiq(package_root):
    # Load the unchanged PyIQA MUSIQ architecture without importing unrelated
    # metrics (CLIP/transformers/etc.) into the restoration environment.
    sys.path.insert(0, str(package_root))
    for name in ('pyiqa', 'pyiqa.archs', 'pyiqa.data', 'pyiqa.utils'):
        package = types.ModuleType(name)
        package.__path__ = [str(package_root.joinpath(*name.split('.')))]
        sys.modules[name] = package
    cls = importlib.import_module('pyiqa.archs.musiq_arch').MUSIQ
    return cls(pretrained='koniq10k').cuda().eval().requires_grad_(False)


@torch.no_grad()
def evaluate(package_root):
    metric = load_musiq(package_root)
    paths = sorted(DATA.glob('*.png'))
    assert len(paths) == 82
    rows = []
    for index, path in enumerate(paths):
        row = {'image': path.name}
        for method in ('input', 'baseline', 'prior', 'rchre'):
            source = path if method == 'input' else IMAGES / method / path.name
            score = float(metric(tensor(source)).item())
            assert np.isfinite(score), (method, path)
            row[method] = score
        rows.append(row)
        print(f'MUSIQ {index + 1}/{len(paths)} {json.dumps(row)}', flush=True)
    delta = np.array([r['rchre'] - r['baseline'] for r in rows])
    rng = np.random.default_rng(20260923)
    boot = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(1)
    report = {
        'metric': 'MUSIQ, PyIQA 0.1.13, KonIQ-10k pretrained, higher is better',
        'preprocessing': 'RGB float [0,1] from saved uint8 PNG; native resolution; built-in MUSIQ multiscale preprocessing',
        'n_images': len(rows),
        'means': {m: float(np.mean([r[m] for r in rows])) for m in ('input', 'baseline', 'prior', 'rchre')},
        'rchre_minus_baseline': {'delta': float(delta.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
                                 'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum())},
        'rows': rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('enhance', 'evaluate'), required=True)
    parser.add_argument('--pyiqa-root', type=Path, default=Path('/tmp/llrw_iqa_deps'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.phase == 'enhance':
        enhance()
    else:
        evaluate(args.pyiqa_root)
