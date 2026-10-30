#!/usr/bin/env python3
"""Evaluate saved RCHRE ablations on official LOL-v1 eval15, never our485."""
import hashlib
import importlib.util
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
from tools.evaluate_rchre_lolv1_five_metrics import (
    LOW, GT, PRIOR, BASELINES, rgb, fid, inception_features)
from tools.evaluate_lsrw_official_checkpoint_fusion import tensor
from tools.train_reliability_residual_gate import ReliabilityResidualGate, apply_gate, ALPHA

OUT = ROOT / 'reproduction_records/rchre_ablation_eval15_five_metrics_20260923'
VARIANTS = {
    'fixed_experts': 'rchre_ablation_fixed_experts',
    'no_acceptance_and_safety': 'rchre_ablation_no_global_acceptance',
    'softmax': 'rchre_ablation_softmax',
    'no_reliability': 'rchre_no_reliability_ablation_v1',
    'rchre': 'rchre_validation_v1',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_variant(name, folder):
    directory = ROOT / 'experiments' / folder
    training_report = json.loads((directory / 'summary.json').read_text())
    protocol = dict(training_report['protocol'])
    legacy = {}
    if name in ('rchre', 'no_reliability'):
        # These two reports predate the structural switches. Validate their
        # actual held-out filenames, rather than trusting only default metadata.
        legacy = {'split_seed': 20260922, 'learnable_experts': True,
                  'global_acceptance': True, 'routing': 'sigmoid'}
        for key, value in legacy.items():
            protocol.setdefault(key, value)
    pool = sorted((ROOT / 'dataset/LOL-v1/our485/low').glob('*.png'))
    assert len(pool) == 485
    heldout = {pool[i].stem for i in np.random.RandomState(20260922).permutation(485)[385:]}
    actual = {Path(r['image']).stem for r in training_report['validation']['rows']['baseline']}
    assert actual == heldout, name
    assert protocol['seed'] == protocol['split_seed'] == 20260922
    assert (protocol['steps'], protocol['train_images'], protocol['heldout_images']) == (4000, 385, 100)
    # Each imported module has independent globals, so ablated inference cannot
    # accidentally use the full model's gates, expert behavior, or routing.
    spec = importlib.util.spec_from_file_location('rchre_eval_' + name, ROOT / 'tools/train_rchre_validation.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    source.LEARNABLE_EXPERTS = protocol['learnable_experts']
    source.USE_GLOBAL_ACCEPTANCE = protocol['global_acceptance']
    source.USE_RELIABILITY = protocol['reliability_supervision']
    source.ROUTING = protocol['routing']
    model = source.RCHRE().cuda().eval().requires_grad_(False)
    checkpoint = torch.load(directory / 'model.pt', map_location='cpu')
    assert checkpoint['seed'] == 20260922 and checkpoint['step'] == 4000
    model.load_state_dict(checkpoint['params'], strict=True)
    return model, {'training_protocol': protocol, 'legacy_protocol_defaults': legacy,
                   'heldout_filenames_verified': True, 'checkpoint': str(directory / 'model.pt'),
                   'sha256': digest(directory / 'model.pt')}


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    paths = sorted(LOW.glob('*.png'))
    assert len(paths) == 15
    # Explicitly verify official test images are not copies of the training pool.
    train_root = ROOT / 'dataset/LOL-v1/our485'
    hashes = {role: {digest(p) for p in (train_root / role).glob('*.png')} for role in ('low', 'high')}
    base = BASELINES['official_checkpoint']
    manifest = []
    for path in paths:
        gt_path = GT / path.name
        assert digest(path) not in hashes['low']
        assert digest(gt_path) not in hashes['high']
        entries = {'low': path, 'gt': gt_path, 'baseline': base / path.name, 'prior': PRIOR / path.name}
        shapes = [rgb(p).shape for p in entries.values()]
        assert len(set(shapes)) == 1, (path, shapes)
        manifest.append({'image': path.name, 'shape': shapes[0],
                         'files': {k: {'path': str(p), 'sha256': digest(p)} for k, p in entries.items()}})
    models, metadata = {}, {}
    for name, folder in VARIANTS.items():
        models[name], metadata[name] = load_variant(name, folder)
    rsra_path = ROOT / 'experiments/reliability_residual_gate_v1/gate.pt'
    rsra = ReliabilityResidualGate().cuda().eval().requires_grad_(False)
    rsra.load_state_dict(torch.load(rsra_path, map_location='cpu')['params'], strict=True)
    perceptual = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    inception = InceptionV3([3], resize_input=True, normalize_input=True).cuda().eval()
    methods = ['baseline', 'prior', 'average', 'learned_scalar', 'rsra', *VARIANTS]
    rows = {m: [] for m in methods}
    images = {m: [] for m in methods}
    OUT.mkdir(parents=True, exist_ok=True)
    for m in methods:
        (OUT / 'images' / m).mkdir(parents=True, exist_ok=True)
    targets = [rgb(GT / p.name) for p in paths]
    for i, (path, target) in enumerate(zip(paths, targets)):
        low, ddm, prior = [tensor(rgb(p)).cuda() for p in (path, base / path.name, PRIOR / path.name)]
        predictions = {'baseline': ddm, 'prior': prior, 'average': (ddm + prior) / 2,
                       'learned_scalar': (1 - ALPHA) * ddm + ALPHA * prior,
                       'rsra': apply_gate(rsra, low, ddm, prior)[0]}
        predictions.update({name: net(low, ddm, prior)[0] for name, net in models.items()})
        target_tensor = tensor(target).cuda()
        for name, prediction in predictions.items():
            image = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            images[name].append(image)
            assert cv2.imwrite(str(OUT / 'images' / name / path.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows[name].append({'image': path.name,
                'psnr': float(calculate_psnr(image, target, crop_border=0)),
                'ssim': float(calculate_ssim(image, target, crop_border=0)),
                'lpips': float(perceptual(tensor(image).cuda() * 2 - 1, target_tensor * 2 - 1).item()),
                'mae': float(np.abs(image.astype(np.float64) - target.astype(np.float64)).mean() / 255)})
        print(f'paired metrics {i+1}/15 {path.name}', flush=True)
    gt_features = inception_features(inception, targets)
    means = {}
    for name in methods:
        means[name] = {k: float(np.mean([r[k] for r in rows[name]])) for k in ('psnr', 'ssim', 'lpips', 'mae')}
        means[name]['fid'] = fid(gt_features, inception_features(inception, images[name]))
        print(name, json.dumps(means[name]), flush=True)
    historical = json.loads((ROOT / 'reproduction_records/rchre_lolv1_five_metrics_20260923/summary.json').read_text())['results']['official_checkpoint']['means']
    checks = {}
    for name in ('baseline', 'learned_scalar', 'rsra', 'rchre'):
        checks[name] = {k: means[name][k] - historical[name][k] for k in means[name]}
        for k, difference in checks[name].items():
            assert abs(difference) < (0.02 if k == 'fid' else 1e-5), (name, k, difference)
    rng = np.random.default_rng(20260923)
    comparisons = {}
    for name in methods:
        if name == 'rchre':
            continue
        comparisons[name] = {}
        for metric in ('psnr', 'ssim', 'lpips', 'mae'):
            delta = np.array([a[metric] - b[metric] for a, b in zip(rows['rchre'], rows[name])])
            boot = delta[rng.integers(0, 15, (20000, 15))].mean(1)
            comparisons[name][metric] = {'full_minus_variant': float(delta.mean()),
                'ci95': np.quantile(boot, [.025, .975]).tolist()}
    report = {'protocol': {
        'evaluation_set': 'LOL-v1 official eval15, 15 pairs; NOT our485 heldout100',
        'backbone': 'public LOL-v1 checkpoint; shared existing seed-0 cached PNGs',
        'prior': 'RetinexFormer LOL_v1.pth shared existing PNGs',
        'module_training': 'RCHRE variants: same our485 split 385 training / 100 validation; seed 20260922; 4000 steps',
        'checkpoint_selection': 'existing frozen final-step checkpoints; no fitting on eval15',
        'metrics': 'full RGB uint8 PSNR/SSIM crop_border=0; full-resolution Alex LPIPS [-1,1]; RGB MAE [0,1]; patched InceptionV3 2048D FID, 15 predictions vs 15 GT',
        'fid_weights': 'pt_inception-2015-12-05-6726825d.pth',
        'scalar_alpha': ALPHA, 'rsra_checkpoint_sha256': digest(rsra_path),
        'limitations': ['eval15 was previously evaluated; not a new blind test', 'FID small sample: auxiliary only',
                       'no_acceptance also removes acceptance supervision and safety loss',
                       'softmax also normalizes oracle routing targets', 'scalar/RSRA are method controls, not identical-training component ablations']},
        'variants': metadata, 'manifest': manifest, 'means': means, 'rows': rows,
        'historical_anchor_deltas': checks, 'full_minus_controls_paired': comparisons}
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DONE: ' + str(OUT / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
