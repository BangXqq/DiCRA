#!/usr/bin/env python3
"""Frozen seed-0 LOLv2 five-metric evaluation of pre-registered RCHRE ablations."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import cv2
import lpips
import numpy as np
from scipy import linalg
import torch
from basicsr.archs.inception import InceptionV3
from basicsr.metrics import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_lsrw_official_checkpoint_fusion import tensor

DIFFUSION_SEEDS = tuple(int(x) for x in os.environ.get('RCHRE_DIFFUSION_SEEDS', '0,1,2').split(','))
OUT = ROOT / os.environ.get('RCHRE_ABLATION_EVAL_OUT',
                            'reproduction_records/rchre_component_ablation_lolv2_3seed_20260923')
SETS = {
    'Real': {'low': ROOT/'dataset/LOLv2_Real/Test/Low', 'gt': ROOT/'dataset/LOLv2_Real/Test/Normal',
             'prior': ROOT/'experiments/retinexformer_lolv2_real_official_prior',
             'base': ROOT/'results/Diff-Retinex_Plus_lolv2_real_test/visualization', 'prefix': 'normal'},
    'Synthetic': {'low': ROOT/'dataset/LOLv2_Synthetic/Test/Low', 'gt': ROOT/'dataset/LOLv2_Synthetic/Test/Normal',
                  'prior': ROOT/'experiments/retinexformer_lolv2_synthetic_official_prior',
                  'base': ROOT/'results/Diff-Retinex_Plus_lolv2_syn_test/visualization', 'prefix': None},
}
VARIANTS = {
    'fixed_experts': ('rchre_ablation_fixed_experts', (1,1,1), False, True, True, 'sigmoid'),
    'no_illumination': ('rchre_ablation_no_illumination', (0,1,1), True, True, True, 'sigmoid'),
    'no_color': ('rchre_ablation_no_color', (1,0,1), True, True, True, 'sigmoid'),
    'no_detail': ('rchre_ablation_no_detail', (1,1,0), True, True, True, 'sigmoid'),
    'softmax': ('rchre_ablation_softmax', (1,1,1), True, True, True, 'softmax'),
    'no_reliability': ('rchre_no_reliability_ablation_v1', (1,1,1), True, True, True, 'sigmoid'),
    'no_safety': ('rchre_ablation_no_safety', (1,1,1), True, True, False, 'sigmoid'),
    'no_acceptance': ('rchre_ablation_no_acceptance_with_safety', (1,1,1), True, False, True, 'sigmoid'),
    'no_acceptance_no_safety': ('rchre_ablation_no_global_acceptance', (1,1,1), True, False, False, 'sigmoid'),
    'full': ('rchre_validation_v1', (1,1,1), True, True, True, 'sigmoid'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(path):
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def load_models():
    models, meta = {}, {}
    for name, (folder, active, learnable, acceptance, safety, routing) in VARIANTS.items():
        spec = importlib.util.spec_from_file_location('rchre_lolv2_' + name, ROOT/'tools/train_rchre_validation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ACTIVE_EXPERTS = active
        module.LEARNABLE_EXPERTS = learnable
        module.USE_GLOBAL_ACCEPTANCE = acceptance
        module.USE_SAFETY_LOSS = safety
        module.ROUTING = routing
        model = module.RCHRE().cuda().eval().requires_grad_(False)
        directory = ROOT/'experiments'/folder
        checkpoint = torch.load(directory/'model.pt', map_location='cpu')
        assert checkpoint['seed'] == 20260922 and checkpoint['step'] == 4000
        model.load_state_dict(checkpoint['params'], strict=True)
        report = json.loads((directory/'summary.json').read_text())
        validation_names = {Path(r['image']).stem for r in report['validation']['rows']['baseline']}
        models[name] = model
        meta[name] = {'folder': folder, 'active_experts': active, 'learnable_experts': learnable,
                      'acceptance': acceptance, 'safety_loss': safety, 'routing': routing,
                      'checkpoint_sha256': digest(directory/'model.pt'),
                      'training_validation_name_sha256': hashlib.sha256('\n'.join(sorted(validation_names)).encode()).hexdigest()}
    assert len({m['training_validation_name_sha256'] for m in meta.values()}) == 1
    return models, meta


def fid(a, b):
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root, _ = linalg.sqrtm(ca @ cb, disp=False)
    if not np.isfinite(root).all():
        offset = np.eye(ca.shape[0]) * 1e-6
        root = linalg.sqrtm((ca + offset) @ (cb + offset))
    if np.iscomplexobj(root):
        root = root.real
    d = ma - mb
    return float(d @ d + np.trace(ca) + np.trace(cb) - 2*np.trace(root))


@torch.no_grad()
def features(net, values):
    result = []
    for i in range(0, len(values), 10):
        result.append(net(torch.cat([tensor(x) for x in values[i:i+10]], 0).cuda())[0].flatten(1).cpu().numpy())
    return np.concatenate(result)


def compare(full, other):
    rng = np.random.default_rng(20260923)
    result = {}
    names = sorted({row['image'] for row in full})
    for key in ('psnr','ssim','lpips','mae'):
        fa = {(r['image'],r['seed']):r[key] for r in full}
        fb = {(r['image'],r['seed']):r[key] for r in other}
        delta = np.array([np.mean([fa[(name,s)]-fb[(name,s)] for s in DIFFUSION_SEEDS])
                          for name in names])
        boot = delta[rng.integers(0, len(delta), (20000,len(delta)))].mean(1)
        result[key] = {'full_minus_ablation': float(delta.mean()),
                       'ci95': np.quantile(boot,[.025,.975]).tolist()}
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    models, model_meta = load_models()
    perceptual = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    inception = InceptionV3([3], resize_input=True, normalize_input=True).cuda().eval()
    report = {'protocol': {
        'preregistered_in': 'reproduction_records/rchre_component_ablation_preregister_20260923.md',
        'sets': 'LOLv2 Real and Synthetic official tests, 100 pairs each',
        'diffusion_seeds': list(DIFFUSION_SEEDS),
        'aggregation': 'means over all image-seed predictions; paired bootstrap clusters diffusion seeds by image; FID calculated per seed then averaged',
        'checkpoint_selection': 'fixed final 4000-step checkpoint',
        'metrics': 'RGB uint8 PSNR/SSIM; full-resolution Alex LPIPS; normalized RGB MAE; patched-InceptionV3 FID',
        'fid_n': 100, 'fid_weights': 'pt_inception-2015-12-05-6726825d.pth',
        'test_tuning': False}, 'variants': model_meta, 'results': {}}
    for split, cfg in SETS.items():
        paths = sorted(cfg['low'].glob('*.png'))
        assert len(paths) == 100
        methods = ['baseline', *VARIANTS]
        rows = {m: [] for m in methods}
        fid_by_seed = {m: [] for m in methods}
        for seed in DIFFUSION_SEEDS:
            images = {m: [] for m in methods}
            targets = []
            suffix = '' if seed == 0 else f'_seed{seed}'
            base_dir = Path(str(cfg['base']).replace('/visualization', suffix + '/visualization'))
            for index, path in enumerate(paths):
                gt_name = cfg['prefix'] + path.name[3:] if cfg['prefix'] else path.name
                target = rgb(cfg['gt']/gt_name)
                low, baseline, prior = [tensor(rgb(p)).cuda() for p in (path, base_dir/path.name, cfg['prior']/path.name)]
                predictions = {'baseline': baseline}
                predictions.update({name: model(low, baseline, prior)[0] for name, model in models.items()})
                target_tensor = tensor(target).cuda()
                targets.append(target)
                for name, prediction in predictions.items():
                    image = (prediction[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
                    images[name].append(image)
                    rows[name].append({'image': path.name, 'seed': seed,
                        'psnr': float(calculate_psnr(image,target,crop_border=0)),
                        'ssim': float(calculate_ssim(image,target,crop_border=0)),
                        'lpips': float(perceptual(tensor(image).cuda()*2-1,target_tensor*2-1).item()),
                        'mae': float(np.abs(image.astype(np.float64)-target.astype(np.float64)).mean()/255)})
                if (index+1) % 20 == 0:
                    print(f'{split} seed {seed} paired {index+1}/100', flush=True)
            gt_features = features(inception, targets)
            for name in methods:
                fid_by_seed[name].append(fid(gt_features, features(inception, images[name])))
        means = {}
        for name in methods:
            means[name] = {key: float(np.mean([r[key] for r in rows[name]])) for key in ('psnr','ssim','lpips','mae')}
            means[name]['fid'] = float(np.mean(fid_by_seed[name]))
            print(split, name, json.dumps(means[name]), flush=True)
        report['results'][split] = {'means': means, 'fid_by_seed': fid_by_seed, 'rows': rows,
            'full_minus_ablation': {name: compare(rows['full'], rows[name]) for name in methods if name != 'full'}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE', OUT/'summary.json', flush=True)


if __name__ == '__main__':
    main()
