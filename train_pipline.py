"""模型训练入口。

该文件负责串联训练所需的各个环节：读取配置、创建数据加载器、恢复训练状态、
构建模型、执行迭代训练与验证，以及保存模型和日志。
"""

import datetime
import logging
import math
import os
import time
import torch
from os import path as osp

from scripts.options import parse_options
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, get_env_info, get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename)
from basicsr.utils.options import copy_opt_file, dict2str


def init_tb_loggers(opt):
    """根据配置初始化 W&B 和 TensorBoard，并返回 TensorBoard 日志器。

    调试实验（名称中含 ``debug``）不会启用这两个在线/可视化日志器，以免产生
    不必要的正式实验记录。
    """
    # W&B 需要先于 TensorBoard 初始化，才能正确同步 TensorBoard 中的指标。
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        # 当前实现通过 TensorBoard 向 W&B 同步数据，因此启用 W&B 时必须同时启用 TensorBoard。
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    """根据配置创建训练集和一个或多个验证集的数据加载器。

    返回值依次为：训练加载器、训练采样器、验证加载器列表、预计总轮数和目标
    总迭代次数。真正控制训练终止的是 ``total_iters``，总轮数只是据此估算。
    """
    train_loader, val_loaders = None, []

    # datasets 中通常包含 train、val，也可以包含 val_xxx 形式的多个验证集。
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # enlarge_ratio 通过重复采样“放大”一个 epoch，不会真的复制数据文件。
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)

            # EnlargedSampler 同时负责数据放大和分布式训练时不同进程之间的数据划分。
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            # DataLoader 在训练阶段使用 drop_last=True，因此必须以它的实际长度为准。
            # 使用 ceil(数据量 / batch) 会比真实值多算一个 batch，进而让有限 epoch
            # 循环在到达 total_iter 前提前结束。
            num_iter_per_epoch = len(train_loader)
            if num_iter_per_epoch == 0:
                raise RuntimeError('Training dataloader contains zero batches. Check the dataset and batch size.')
            total_iters = int(opt['train']['total_iter'])
            # total_epochs 只用于展示估算；训练结束只由 current_iter 控制。
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            # 验证时不打乱、不放大数据，因此不需要训练采样器。
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def _rooted_path(opt, path):
    """将配置中的相对路径稳定地解析到项目根目录。"""
    path = osp.expanduser(path)
    if not osp.isabs(path):
        path = osp.join(opt['root_path'], path)
    return osp.abspath(path)


def _state_iteration_from_name(state_path):
    """从 ``<iter>.state`` 文件名提取迭代数。"""
    filename = osp.basename(state_path)
    if not filename.endswith('.state'):
        raise ValueError(f'Not a training-state file: {state_path}')
    try:
        return int(filename[:-len('.state')])
    except ValueError as exc:
        raise ValueError(f'Training-state filename must be <iter>.state: {state_path}') from exc


def _matching_model_candidates(opt, state_path, iteration, explicit_pair=False):
    """返回与 state 同迭代的项目专用 DDPM 权重候选路径。

    YAML 中的模型路径只允许和 YAML 中显式指定的 state 配对。自动扫描当前
    实验时必须使用 state 所在实验目录中的同迭代模型，避免两个旧格式断点
    （尚无 checkpoint_id）仅因迭代数相同而被跨实验误配。
    """
    filename = f'net_g_ddpm_{iteration}.pth'
    candidates = []

    configured_model = opt['path'].get('pretrain_network_ddpm') if explicit_pair else None
    if configured_model:
        configured_model = _rooted_path(opt, configured_model)
        if osp.basename(configured_model) != filename:
            return [configured_model]
        # 显式 model/state 是一个整体；路径错误时应报错而非静默换用其他模型。
        return [configured_model]

    # 自动恢复只在 state 自己所属的实验目录内寻找配对权重。
    state_experiment_root = osp.dirname(osp.dirname(state_path))
    candidates.append(osp.join(state_experiment_root, 'models', filename))

    # 按顺序去重。
    return list(dict.fromkeys(osp.abspath(path) for path in candidates))


def _validate_resume_pair(state_path, model_path, expected_iteration):
    """在 CPU 上完整读取并校验一对恢复文件，避免选择半写入断点。"""
    if not osp.isfile(state_path):
        raise FileNotFoundError(f'Resume state does not exist: {state_path}')
    if not osp.isfile(model_path):
        raise FileNotFoundError(f'Matching model does not exist: {model_path}')
    expected_model_name = f'net_g_ddpm_{expected_iteration}.pth'
    if osp.basename(model_path) != expected_model_name:
        raise ValueError(
            f'Matching model filename must be {expected_model_name}: {model_path}')

    state = torch.load(state_path, map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError(f'Resume state is not a dictionary: {state_path}')
    required_state_keys = {'epoch', 'iter', 'optimizers', 'schedulers'}
    missing_state_keys = required_state_keys.difference(state)
    if missing_state_keys:
        raise ValueError(f'Resume state is missing keys {sorted(missing_state_keys)}: {state_path}')
    if int(state['iter']) != int(expected_iteration):
        raise ValueError(
            f'Resume-state iteration mismatch: filename={expected_iteration}, payload={state["iter"]}: {state_path}')

    model = torch.load(model_path, map_location='cpu')
    if not isinstance(model, dict) or not isinstance(model.get('params'), dict):
        raise ValueError(f'Model checkpoint does not contain a params dictionary: {model_path}')
    model_metadata = model.get('metadata')
    if isinstance(model_metadata, dict) and model_metadata.get('iter') is not None:
        if int(model_metadata['iter']) != int(expected_iteration):
            raise ValueError(
                f'Model-checkpoint iteration mismatch: expected={expected_iteration}, '
                f'payload={model_metadata["iter"]}: {model_path}')
    state_checkpoint_id = state.get('checkpoint_id')
    model_checkpoint_id = model.get('checkpoint_id')
    if (state_checkpoint_id is None) != (model_checkpoint_id is None):
        raise ValueError(
            f'Model/state checkpoint generation metadata differs: model={model_checkpoint_id}, '
            f'state={state_checkpoint_id}')
    if state_checkpoint_id is not None and state_checkpoint_id != model_checkpoint_id:
        raise ValueError(
            f'Model/state checkpoint generation mismatch: model={model_checkpoint_id}, state={state_checkpoint_id}')
    del model
    del state


def resolve_resume_checkpoint(opt, announce=True):
    """解析最新的有效 model/state 对，并更新配置中的两条恢复路径。

    自动恢复会按迭代数从大到小检查所有 ``.state``，跳过损坏文件或缺少
    同迭代 ``net_g_ddpm_<iter>.pth`` 的孤立文件。显式恢复则严格校验指定
    state，任何错配都会直接报错，绝不带着旧权重继续训练。
    """
    explicit_state = opt['path'].get('resume_state')
    invalid_candidates = []

    if opt['auto_resume']:
        state_dir = opt['path']['training_states']
        state_candidates = []
        if osp.isdir(state_dir):
            filenames = os.listdir(state_dir)
        else:
            filenames = []
        for filename in filenames:
            if not filename.endswith('.state'):
                continue
            state_path = osp.join(state_dir, filename)
            try:
                iteration = _state_iteration_from_name(state_path)
            except ValueError as exc:
                invalid_candidates.append(str(exc))
                continue
            state_candidates.append((iteration, osp.abspath(state_path), False))

        # --auto_resume 仍保留 YAML 显式断点作为候选。这样新实验可以从外部
        # 成对断点启动，后续重启又能自动选择当前实验目录中的更高迭代断点。
        if explicit_state:
            explicit_state_path = _rooted_path(opt, explicit_state)
            try:
                explicit_iteration = _state_iteration_from_name(explicit_state_path)
                state_candidates.append((explicit_iteration, explicit_state_path, True))
            except ValueError as exc:
                invalid_candidates.append(str(exc))

        # 同一路径若既被扫描到又是 YAML 显式 state，保留“显式配对”属性。
        deduplicated = {}
        for iteration, state_path, explicit_pair in state_candidates:
            previous = deduplicated.get(state_path)
            deduplicated[state_path] = (
                iteration,
                state_path,
                explicit_pair or (previous[2] if previous else False))
        state_candidates = list(deduplicated.values())
        # 高迭代优先；同迭代时先尝试当前实验自动候选，再尝试外部显式候选。
        state_candidates.sort(key=lambda item: (item[0], not item[2]), reverse=True)
    elif explicit_state:
        state_path = _rooted_path(opt, explicit_state)
        # 显式路径的文件名必须和 state 内部迭代数一致。
        state_candidates = [(_state_iteration_from_name(state_path), state_path, True)]
    else:
        return None, None, None

    for iteration, state_path, explicit_pair in state_candidates:
        errors = []
        for model_path in _matching_model_candidates(
                opt, state_path, iteration, explicit_pair=explicit_pair):
            try:
                _validate_resume_pair(state_path, model_path, iteration)
            # torch.load 对不同损坏形式可能抛出 EOFError、UnpicklingError 等；
            # 自动恢复应跳过任何不可完整读取的候选，但不捕获 KeyboardInterrupt/SystemExit。
            except Exception as exc:
                errors.append(str(exc))
                continue

            opt['path']['resume_state'] = state_path
            opt['path']['pretrain_network_ddpm'] = model_path
            if announce:
                print(
                    f'Use validated resume pair at iter {iteration}:\n'
                    f'  model: {model_path}\n  state: {state_path}',
                    flush=True)
            return state_path, model_path, iteration

        invalid_candidates.append(f'iter {iteration}: ' + ' | '.join(errors))
        if not opt['auto_resume']:
            break

    if state_candidates:
        details = '\n  - '.join(invalid_candidates)
        raise RuntimeError(f'No valid matching model/state checkpoint pair was found:\n  - {details}')
    if invalid_candidates:
        details = '\n  - '.join(invalid_candidates)
        raise RuntimeError(f'No usable training-state filename was found:\n  - {details}')
    if opt['auto_resume']:
        model_dir = opt['path']['models']
        if osp.isdir(model_dir):
            orphan_models = [
                filename for filename in os.listdir(model_dir)
                if filename.startswith('net_g_ddpm_') and filename.endswith('.pth')]
            if orphan_models:
                raise RuntimeError(
                    'Auto-resume found existing DDPM model artifacts but no training state; '
                    'refusing to restart silently from iter 0. Inspect: '
                    + ', '.join(sorted(orphan_models)[:5]))
    return None, None, None


def load_resume_state(opt):
    """查找并加载断点文件，未配置断点续训时返回 ``None``。

    开启自动恢复时，会从当前实验中选择迭代数最大的完整 model/state 对；
    否则严格校验配置中显式指定的 state 及其同迭代模型。损坏、缺失或迭代数
    错配的文件不会被用于恢复。

    ``.state`` 主要保存 epoch、迭代数、优化器和调度器状态，并不保存数据
    加载器正处于哪个 batch，因此恢复后会从记录的 epoch 开头重新取数据。
    """
    resume_state_path, _, _ = resolve_resume_checkpoint(opt)
    if resume_state_path is None:
        return None

    # GPU 训练保持原项目的加载行为；CPU 模式则显式加载到 CPU。
    if opt['num_gpu'] == 0:
        return torch.load(resume_state_path, map_location='cpu')
    device_id = torch.cuda.current_device()
    return torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))


def train_pipeline(root_path):
    """执行完整的训练流程。

    Args:
        root_path (str): 项目根目录，通常由同目录下的 ``train.py`` 传入。
    """
    # 解析配置，同时完成分布式环境和随机种子的初始化。
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    # 对输入尺寸相对固定的卷积网络，benchmark 通常能选择更快的 cuDNN 算法；
    # 但它可能降低结果的严格可复现性，如需复现可结合 deterministic 配置调整。
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 如果配置了断点续训，先读取上次保存的训练状态。
    resume_state = load_resume_state(opt)

    # 新实验需要创建目录；恢复实验时保留原目录，避免重命名或覆盖已有结果。
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # 保存本次使用的 YAML 配置，方便之后复现实验。
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # 注意：在此之前（包括上面调用的函数内部）不要调用 get_root_logger，
    # 否则根日志器会被提前创建，导致这里指定的日志文件无法正确生效。
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # 初始化实验指标的可视化日志器。
    tb_logger = init_tb_loggers(opt)

    # 创建训练和验证数据加载器，并计算训练所需的 epoch/iteration 数量。
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # build_model 会根据 opt 中的 model_type 创建网络、损失、优化器和学习率调度器。
    model = build_model(opt)
    if resume_state:
        # 模型权重已由 build_model 按配置加载；这里继续恢复优化器和学习率调度器。
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # MessageLogger 负责格式化损失、学习率、耗时和预计剩余时间等训练信息。
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 该封装统一提供 reset()/next() 接口；配置 CPU/CUDA 预取时可减少等待数据的时间。
    # prefetch_mode=None 时仍使用 CPUPrefetcher 外壳，但底层是普通 DataLoader。
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        # CUDA 预取可异步将数据搬到 GPU，但依赖 DataLoader 开启 pin_memory。
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # 训练由 current_iter 驱动；epoch 主要用于重置采样器和完整遍历数据集。
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    if current_iter > total_iters:
        raise RuntimeError(f'Resume iteration {current_iter} exceeds configured total_iter {total_iters}.')

    epoch = start_epoch
    while current_iter < total_iters:
        # 将 epoch 编号交给采样器，以改变本轮的确定性乱序并协调各进程的数据分片。
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None and current_iter < total_iters:
            # 近似记录等待当前批次数据所花费的时间。
            data_timer.record()
            current_iter += 1

            # 先按当前迭代数更新学习率；warmup_iter=-1 表示不启用预热。
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # 将当前批次交给模型，并完成前向传播、反向传播和参数更新。
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()

            if current_iter == 1:
                # 首次迭代可能包含初始化开销，重置起点可让 ETA 更准确；
                # 断点续训时 current_iter 不会等于 1，因此不会执行这里。
                msg_logger.reset_start_time()

            # 按 print_freq 汇总并输出当前训练指标。
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # 到达验证间隔时，依次在所有验证集上评估模型。
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            # 验证先于断点保存，使本轮更新后的最佳指标也进入 training state。
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # 从此处重新开始计时，并预取下一批数据供下一次循环使用。
            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

        epoch += 1

    if current_iter != total_iters:
        raise RuntimeError(f'Training stopped at iter {current_iter}, expected exactly {total_iters}.')

    # 退出训练循环后，统计总耗时并保存“latest”模型。
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    # BasicSR 约定 -1 表示使用 latest 作为保存标识，而不是实际的 epoch/iteration。
    model.save(epoch=-1, current_iter=-1)

    # 训练结束后再完整验证一次最终模型。
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

    # 将缓存中的 TensorBoard 事件写入磁盘并释放资源。
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    # 正常训练入口是同目录下的 train.py，它会正确传入仓库根目录。
    # 若直接运行本文件，下面的两级上移会得到仓库的上一级目录。
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
