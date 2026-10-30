#!/usr/bin/env python3
"""在启动长时间训练前执行只读完整性与资源检查。"""

import argparse
from collections.abc import Mapping
from pathlib import Path
import os
import shutil
import subprocess
import sys

import cv2
import torch
import yaml


# LoLDataset 当前只会按此顺序 glob 小写扩展名；预检必须和真实 loader 一致。
LOADER_IMAGE_SUFFIXES = ('.png', '.jpg')
IMAGE_LIKE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _resolved_path(project_root, value):
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _loader_image_paths(path):
    paths = []
    for suffix in LOADER_IMAGE_SUFFIXES:
        paths.extend(sorted(path.glob(f'*{suffix}')))
    return paths


def _check_datasets(project_root, config, errors, summary):
    for phase, dataset in config.get('datasets', {}).items():
        gt_value = dataset.get('gt_root')
        input_value = dataset.get('input_root')
        if not gt_value or not input_value:
            errors.append(f'datasets.{phase} must define both gt_root and input_root.')
            continue

        gt_path = _resolved_path(project_root, gt_value)
        input_path = _resolved_path(project_root, input_value)
        for label, path in (('gt_root', gt_path), ('input_root', input_path)):
            if not path.is_dir():
                errors.append(f'datasets.{phase}.{label} is missing: {path}')

        if not gt_path.is_dir() or not input_path.is_dir():
            continue

        ignored_gt = sorted(
            item.name for item in gt_path.iterdir()
            if item.is_file()
            and item.suffix.lower() in IMAGE_LIKE_SUFFIXES
            and item.suffix not in LOADER_IMAGE_SUFFIXES)
        if ignored_gt:
            errors.append(
                f'datasets.{phase}.gt_root has {len(ignored_gt)} image files ignored by LoLDataset '
                f'(only lowercase .png/.jpg are loaded); examples: {ignored_gt[:5]}')

        gt_paths = _loader_image_paths(gt_path)
        if not gt_paths:
            errors.append(f'datasets.{phase}.gt_root contains no loader-supported images: {gt_path}')
            continue

        missing_inputs = []
        unreadable = []
        shape_mismatches = []
        too_small = []
        crop_size = dataset.get('crop_size') if phase.split('_')[0] == 'train' else None

        for gt_file in gt_paths:
            input_file = input_path / gt_file.name
            if not input_file.is_file():
                missing_inputs.append(gt_file.name)
                continue

            gt_image = cv2.imread(str(gt_file), cv2.IMREAD_COLOR)
            input_image = cv2.imread(str(input_file), cv2.IMREAD_COLOR)
            if gt_image is None or input_image is None:
                unreadable.append(gt_file.name)
                continue
            if gt_image.shape != input_image.shape:
                shape_mismatches.append(
                    f'{gt_file.name}: GT{gt_image.shape} vs input{input_image.shape}')
                continue
            if crop_size is not None:
                crop_size = int(crop_size)
                height, width = input_image.shape[:2]
                if height < crop_size or width < crop_size:
                    too_small.append(f'{gt_file.name}: {width}x{height} < crop {crop_size}')

        if missing_inputs:
            errors.append(
                f'datasets.{phase} is missing {len(missing_inputs)} exact-name paired inputs; '
                f'examples: {missing_inputs[:5]}')
        if unreadable:
            errors.append(
                f'datasets.{phase} has {len(unreadable)} unreadable GT/input pairs; examples: {unreadable[:5]}')
        if shape_mismatches:
            errors.append(
                f'datasets.{phase} has {len(shape_mismatches)} shape-mismatched pairs; '
                f'examples: {shape_mismatches[:3]}')
        if too_small:
            errors.append(
                f'datasets.{phase} has {len(too_small)} images smaller than crop_size; examples: {too_small[:3]}')

        invalid_count = len(missing_inputs) + len(unreadable) + len(shape_mismatches) + len(too_small)
        summary.append(
            f'dataset {phase}: {len(gt_paths) - invalid_count}/{len(gt_paths)} decodable shape-matched pairs')


def _check_decomposition_weight(project_root, config, errors, summary):
    network_config = config.get('network_decom', {})
    decom_path_value = network_config.get('path')
    if not decom_path_value:
        errors.append('network_decom.path is not configured.')
        return
    decom_path = _resolved_path(project_root, decom_path_value)
    if not decom_path.is_file():
        errors.append(f'Decomposition checkpoint is missing: {decom_path}')
        return

    try:
        checkpoint = torch.load(str(decom_path), map_location='cpu')
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get('model'), Mapping):
            raise ValueError('checkpoint must contain a model state-dict mapping')
        if not checkpoint['model']:
            raise ValueError('model state dict is empty')

        # 使用和正式训练相同的 registry/builder，并以 strict=True 校验全部键和形状。
        import archs  # noqa: F401  # 注册项目网络结构
        from basicsr.archs import build_network
        decomposition_network = build_network(dict(network_config))
        decomposition_network.load_state_dict(checkpoint['model'], strict=True)
        tensor_count = len(checkpoint['model'])
        del decomposition_network
        del checkpoint
    except Exception as exc:
        errors.append(f'Decomposition checkpoint is incompatible or corrupt: {decom_path}: {exc}')
        return

    summary.append(f'decomposition checkpoint: strict-load verified ({tensor_count} tensors)')


def _runtime_resume_options(project_root, config, auto_resume):
    experiment_root = project_root / 'experiments' / str(config['name'])
    path_config = dict(config.get('path', {}))
    path_config.update({
        'experiments_root': str(experiment_root),
        'models': str(experiment_root / 'models'),
        'training_states': str(experiment_root / 'training_states'),
        'log': str(experiment_root),
        'visualization': str(experiment_root / 'visualization'),
    })
    return {
        'root_path': str(project_root),
        'auto_resume': bool(auto_resume),
        'path': path_config,
    }


def _check_resume_pair(project_root, config, auto_resume, errors, summary):
    try:
        from train_pipline import resolve_resume_checkpoint
        runtime_options = _runtime_resume_options(project_root, config, auto_resume)
        state_path, model_path, iteration = resolve_resume_checkpoint(runtime_options, announce=False)
    except Exception as exc:
        errors.append(f'No runtime-valid resume checkpoint pair: {exc}')
        return None

    if state_path is None:
        summary.append('resume: starting without optimizer/scheduler state')
        return None

    summary.append(f'validated resume pair: iter {iteration}')
    summary.append(f'  model: {model_path}')
    summary.append(f'  state: {state_path}')
    return iteration


def _run_nvidia_smi(arguments):
    return subprocess.run(
        ['nvidia-smi', *arguments],
        check=True,
        capture_output=True,
        text=True)


def _check_gpu(gpu_index, errors, summary):
    try:
        inventory = _run_nvidia_smi([
            '--query-gpu=index,uuid,name',
            '--format=csv,noheader,nounits',
        ])
        selected_gpu = None
        for line in inventory.stdout.splitlines():
            fields = [field.strip() for field in line.split(',', 2)]
            if len(fields) == 3 and fields[0] == str(gpu_index):
                selected_gpu = fields
                break
        if selected_gpu is None:
            raise ValueError(f'physical GPU index {gpu_index} does not exist')
        _, selected_uuid, selected_name = selected_gpu

        processes = _run_nvidia_smi([
            '--query-compute-apps=gpu_uuid,pid,process_name',
            '--format=csv,noheader,nounits',
        ])
        active_processes = []
        for line in processes.stdout.splitlines():
            fields = [field.strip() for field in line.split(',', 2)]
            if len(fields) == 3 and fields[0] == selected_uuid:
                active_processes.append(f'pid={fields[1]} {fields[2]}')
        if active_processes:
            errors.append(
                f'GPU {gpu_index} already has compute processes: ' + '; '.join(active_processes))
        else:
            summary.append(f'GPU {gpu_index} compute process check: idle')
        summary.append(f'physical CUDA device {gpu_index}: {selected_name} ({selected_uuid})')
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        errors.append(f'nvidia-smi GPU {gpu_index} check failed: {exc}')
        return

    # nvidia-smi 的进程查询先执行，避免本预检创建的 CUDA context 被误判为占用者。
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        errors.append(
            'The managed process must see exactly one CUDA device; check CUDA_VISIBLE_DEVICES and the driver.')
        return
    try:
        summary.append(f'PyTorch visible device 0: {torch.cuda.get_device_name(0)}')
    except Exception as exc:
        errors.append(f'PyTorch cannot initialize the selected CUDA device: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', required=True, help='Training YAML path.')
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--min-free-gb', type=float, default=20.0)
    parser.add_argument('--gpu-index', type=int, default=0, help='Physical GPU index used by the manager.')
    parser.add_argument('--auto-resume', action='store_true', help='Use the same auto-resume mode as train.py.')
    parser.add_argument('--skip-gpu-check', action='store_true', help='Only for offline/dry-run validation.')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    config_path = _resolved_path(project_root, args.opt)
    errors = []
    summary = [f'project: {project_root}', f'config: {config_path}']

    if not config_path.is_file():
        print(f'NOT READY\n- Training config is missing: {config_path}', file=sys.stderr)
        return 2
    try:
        with config_path.open('r', encoding='utf-8') as file_handle:
            config = yaml.safe_load(file_handle)
        if not isinstance(config, Mapping):
            raise ValueError('top-level YAML value is not a mapping')
        if not config.get('name'):
            raise ValueError('name is missing')
    except Exception as exc:
        print(f'NOT READY\n- Training config cannot be parsed: {config_path}: {exc}', file=sys.stderr)
        return 2

    # 先完整校验恢复 pair 并判断是否已经结束。已完成的 enabled unit 在下次
    # 登录/开机时应干净退出，不应再依赖数据集、GPU 或剩余磁盘是否仍可用。
    resume_iteration = _check_resume_pair(project_root, config, args.auto_resume, errors, summary)

    complete_message = None
    try:
        total_iter = int(config.get('train', {}).get('total_iter', 0))
    except (TypeError, ValueError):
        total_iter = 0
    if total_iter <= 0:
        errors.append(f'Invalid train.total_iter: {total_iter}')
    elif resume_iteration is not None:
        if resume_iteration > total_iter:
            errors.append(
                f'Resume iteration overshoots target: resume={resume_iteration}, total={total_iter}')
        elif resume_iteration == total_iter:
            complete_message = f'experiment already complete: iter {resume_iteration}/{total_iter}'
        else:
            summary.append(f'remaining iterations: {total_iter - resume_iteration}')

    if complete_message is not None and not errors:
        print('COMPLETE')
        for item in summary:
            print(f'- {item}')
        print(f'- {complete_message}')
        # systemd unit 把 10 声明为成功，从而在重启后不会重复训练已完成实验。
        return 10

    configured_gpu_count = config.get('num_gpu')
    if configured_gpu_count not in (1, 'auto'):
        errors.append(
            f'Managed single-GPU runner requires num_gpu: 1 (or auto), got: {configured_gpu_count}')

    free_gib = shutil.disk_usage(project_root).free / (1024**3)
    summary.append(f'free disk: {free_gib:.2f} GiB (required: {args.min_free_gb:.2f} GiB)')
    if free_gib < args.min_free_gb:
        errors.append(f'Insufficient free disk: {free_gib:.2f} GiB < {args.min_free_gb:.2f} GiB')

    _check_datasets(project_root, config, errors, summary)
    _check_decomposition_weight(project_root, config, errors, summary)

    if args.skip_gpu_check:
        summary.append('GPU check: skipped by explicit dry-run flag')
    else:
        _check_gpu(args.gpu_index, errors, summary)

    if errors:
        print('NOT READY', file=sys.stderr)
        for error in errors:
            print(f'- {error}', file=sys.stderr)
        print('\nChecks completed before failure:', file=sys.stderr)
        for item in summary:
            print(f'- {item}', file=sys.stderr)
        return 1

    print('READY')
    for item in summary:
        print(f'- {item}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
