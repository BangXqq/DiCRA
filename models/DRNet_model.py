from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from copy import deepcopy
import json
import math
import numpy as np
import re
import torch.nn as nn
import torch.nn.functional
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F
import os.path as osp
from tqdm import tqdm
import torch
import cv2
import os
import time
import uuid
cv2.setNumThreads(1)
from scripts.utils import pad_tensor_back


def _atomic_torch_save(payload, save_path, required_keys, retries=3):
    """在目标目录中写临时文件，校验后原子替换正式断点。"""
    logger = get_root_logger()
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    last_error = None

    for attempt in range(1, retries + 1):
        temp_path = f'{save_path}.tmp-{os.getpid()}-{attempt}'
        try:
            with open(temp_path, 'wb') as file_handle:
                torch.save(payload, file_handle)
                file_handle.flush()
                os.fsync(file_handle.fileno())

            # 在替换旧断点前完整读回一次；半写入或 ZIP/CRC 损坏会在这里报错。
            reloaded = torch.load(temp_path, map_location='cpu')
            if not isinstance(reloaded, dict):
                raise ValueError('Serialized checkpoint is not a dictionary.')
            missing_keys = set(required_keys).difference(reloaded)
            if missing_keys:
                raise ValueError(f'Serialized checkpoint is missing keys: {sorted(missing_keys)}')
            del reloaded

            os.replace(temp_path, save_path)
            directory_fd = os.open(osp.dirname(save_path), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                f'Atomic checkpoint save failed ({attempt}/{retries}) for {save_path}: {exc}')
            if attempt < retries:
                time.sleep(1)
        finally:
            if osp.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    raise IOError(f'Cannot save a verified checkpoint after {retries} attempts: {save_path}') from last_error


def _atomic_json_save(payload, save_path):
    """原子写入小型 JSON 元数据。"""
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    temp_path = f'{save_path}.tmp-{os.getpid()}'
    try:
        with open(temp_path, 'w', encoding='utf-8') as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write('\n')
            file_handle.flush()
            os.fsync(file_handle.fileno())
        with open(temp_path, 'r', encoding='utf-8') as file_handle:
            if json.load(file_handle) != payload:
                raise ValueError(f'JSON read-back verification failed: {temp_path}')
        os.replace(temp_path, save_path)
        directory_fd = os.open(osp.dirname(save_path), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if osp.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass



@MODEL_REGISTRY.register()
class DRNetModel(BaseModel):
    def __init__(self, opt):
        super(DRNetModel, self).__init__(opt)

        # define sr3-u-net network
        self.sr3unet = build_network(opt['network_sr3unet'])
        self.sr3unet = self.model_to_device(self.sr3unet)
        opt['network_ddpm']['denoise_fn'] = self.sr3unet

        # define consistent-u-net network
        self.consistent_unet = build_network(opt['network_consistentunet'])
        self.consistent_unet = self.model_to_device(self.consistent_unet)
        opt['network_ddpm']['restore_fn'] = self.consistent_unet

        self.ddpm = build_network(opt['network_ddpm'])
        self.ddpm = self.model_to_device(self.ddpm)

        self.decom_net = build_network(opt['network_decom'])
        self.decom_net = self.model_to_device(self.decom_net)

        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_ddpm_model = self.ddpm.module
        else:
            self.bare_ddpm_model = self.ddpm

        self.bare_ddpm_model.set_new_noise_schedule(schedule_opt=opt['ddpm_schedule'], device=self.device)
        self.bare_ddpm_model.set_loss(device=self.device)
        self.print_network(self.ddpm)

        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

        if self.is_train:
            self.init_training_settings()

        logger = get_root_logger()
        is_train_mode = not self.opt['val'].get('test_flag', False)
        if is_train_mode:
            assert os.path.exists(self.opt['network_decom']['path']), logger.info("The decom-network weights is not exsit.")
            load_decom_path = self.opt['network_decom']['path']
            if isinstance(self.decom_net, (DataParallel, DistributedDataParallel)):
                logger.info(self.decom_net.module.load_state_dict(torch.load(load_decom_path, map_location='cuda')['model'], strict=True))
            else:
                logger.info(self.decom_net.load_state_dict(torch.load(load_decom_path, map_location='cuda')['model'], strict=True))
            logger.info("Load the decom-net weight success! Setting --------> eval mode")
        else:
            logger.info("Begin to eval the model!")

        if isinstance(self.decom_net, (DataParallel, DistributedDataParallel)):
            self.decom_net.module.requires_grad_(False)
            self.decom_net.module.eval()
        else:
            self.decom_net.requires_grad_(False)
            self.decom_net.eval()

        load_path = self.opt['path'].get('pretrain_network_ddpm', None)
        if load_path is not None:
            if isinstance(self.decom_net, (DataParallel, DistributedDataParallel)):
                param_key = self.opt['path'].get('param_key_ddpm', 'params')
                self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_ddpm', True), param_key)
            else:
                param_key = self.opt['path'].get('param_key_ddpm', 'params')
                self.load_bare_network(self.bare_ddpm_model, load_path, self.opt['path'].get('strict_load_ddpm', True), param_key)

    def init_training_settings(self):
        self.ddpm.train()

        self.best_val_psnr = 0.0
        self.best_val_psnr_iter = -1
        self.best_val_checkpoint = None
        self.best_val_checkpoint_id = None
        self._restore_best_val_psnr()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    @property
    def _best_metric_path(self):
        return osp.join(self.opt['path']['experiments_root'], 'best_metrics.json')

    def _restore_best_val_psnr(self):
        """从元数据或历史日志恢复跨进程的最佳 PSNR。"""
        best_value = 0.0
        best_iter = -1

        if osp.isfile(self._best_metric_path):
            try:
                with open(self._best_metric_path, 'r', encoding='utf-8') as file_handle:
                    record = json.load(file_handle)
                record_value = float(record.get('psnr', 0.0))
                record_iter = int(record.get('iter', -1))
                record_checkpoint = record.get('best_checkpoint')
                record_checkpoint_id = record.get('checkpoint_id')
                if not math.isfinite(record_value) or record_iter < 0:
                    raise ValueError('best metric value/iteration is invalid')
                if not record_checkpoint:
                    raise ValueError('best_checkpoint is missing')
                self._validate_best_checkpoint(
                    record_checkpoint,
                    checkpoint_id=record_checkpoint_id,
                    expected_psnr=record_value,
                    expected_iter=record_iter)
                best_value = record_value
                best_iter = record_iter
                self.best_val_checkpoint = record_checkpoint
                self.best_val_checkpoint_id = record_checkpoint_id
            except Exception as exc:
                get_root_logger().warning(f'Cannot read best-metric metadata; falling back to logs: {exc}')

        # 兼容旧断点：旧 .state 没有最佳指标，因此从同一实验的既有日志恢复。
        log_dir = self.opt['path'].get('log')
        log_candidates = []
        if log_dir and osp.isdir(log_dir):
            iter_pattern = re.compile(r'\biter:\s*([0-9,]+)')
            psnr_pattern = re.compile(r'#\s*psnr:\s*([0-9]+(?:\.[0-9]+)?)')
            for filename in sorted(os.listdir(log_dir)):
                if not (filename.startswith('train_') and filename.endswith('.log')):
                    continue
                current_iter = -1
                log_path = osp.join(log_dir, filename)
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='replace') as file_handle:
                        for line in file_handle:
                            iter_match = iter_pattern.search(line)
                            if iter_match:
                                current_iter = int(iter_match.group(1).replace(',', ''))
                            psnr_match = psnr_pattern.search(line)
                            if psnr_match:
                                value = float(psnr_match.group(1))
                                if math.isfinite(value) and current_iter >= 0:
                                    ordinary_checkpoint = osp.join(
                                        self.opt['path']['models'], f'net_g_ddpm_{current_iter}.pth')
                                    log_candidates.append((value, current_iter, ordinary_checkpoint))
                except OSError as exc:
                    get_root_logger().warning(f'Cannot scan validation history {log_path}: {exc}')

        # 从高到低只接受确实存在且能完整加载的普通权重，保证指标与权重指针一致。
        for value, candidate_iter, ordinary_checkpoint in sorted(log_candidates, reverse=True):
            if value <= best_value:
                break
            relative_checkpoint = osp.relpath(
                ordinary_checkpoint, self.opt['path']['experiments_root'])
            try:
                self._validate_best_checkpoint(relative_checkpoint)
            except Exception as exc:
                get_root_logger().warning(
                    f'Ignore validation record without a valid checkpoint at iter {candidate_iter}: {exc}')
                continue
            best_value = value
            best_iter = candidate_iter
            self.best_val_checkpoint = relative_checkpoint
            self.best_val_checkpoint_id = None
            break

        self.best_val_psnr = best_value
        self.best_val_psnr_iter = best_iter
        if best_iter >= 0:
            get_root_logger().info(
                f'Restored historical best validation PSNR {best_value:.4f} at iter {best_iter}.')

    def _validate_best_checkpoint(
            self,
            checkpoint_reference,
            checkpoint_id=None,
            expected_psnr=None,
            expected_iter=None):
        """完整读取最佳权重，并校验它和指标元数据属于同一次发布。"""
        checkpoint_path = checkpoint_reference
        if not osp.isabs(checkpoint_path):
            checkpoint_path = osp.join(self.opt['path']['experiments_root'], checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('params'), dict):
            raise ValueError(f'best checkpoint has no params dictionary: {checkpoint_path}')
        if checkpoint_id is not None:
            if checkpoint.get('checkpoint_id') != checkpoint_id:
                raise ValueError('best checkpoint generation does not match metadata')
            metadata = checkpoint.get('metadata')
            if not isinstance(metadata, dict):
                raise ValueError('best checkpoint has no generation metadata')
            if expected_iter is not None and int(metadata.get('best_val_psnr_iter', -1)) != int(expected_iter):
                raise ValueError('best checkpoint iteration does not match metadata')
            if expected_psnr is not None and not math.isclose(
                    float(metadata.get('best_val_psnr', float('nan'))),
                    float(expected_psnr),
                    rel_tol=0.0,
                    abs_tol=1e-12):
                raise ValueError('best checkpoint PSNR does not match metadata')
        del checkpoint

    @master_only
    def _save_best_metric_record(self):
        _atomic_json_save(
            {
                'psnr': float(self.best_val_psnr),
                'iter': int(self.best_val_psnr_iter),
                'best_checkpoint': self.best_val_checkpoint,
                'checkpoint_id': self.best_val_checkpoint_id
            }, self._best_metric_path)

    def _save_new_best(self, value, current_iter):
        """先保存按迭代数和发布 ID 命名的不可变权重，再原子发布指标元数据。"""
        checkpoint_id = uuid.uuid4().hex
        checkpoint_tag = f'{current_iter}_{checkpoint_id}'
        self.save_network(
            [self.ddpm],
            'net_g_best_ddpm_val',
            checkpoint_tag,
            param_key=['params'],
            checkpoint_id=checkpoint_id,
            checkpoint_metadata={
                'best_val_psnr': float(value),
                'best_val_psnr_iter': int(current_iter)
            })
        self.best_val_psnr = float(value)
        self.best_val_psnr_iter = int(current_iter)
        self.best_val_checkpoint = f'models/net_g_best_ddpm_val_{checkpoint_tag}.pth'
        self.best_val_checkpoint_id = checkpoint_id
        self._save_best_metric_record()

    def resume_training(self, resume_state):
        """恢复优化器/调度器，并合并 state 中持久化的最佳指标。"""
        super().resume_training(resume_state)
        state_best = float(resume_state.get('best_val_psnr', 0.0))
        if state_best > self.best_val_psnr:
            state_best_iter = int(resume_state.get('best_val_psnr_iter', -1))
            state_best_checkpoint = resume_state.get('best_val_checkpoint')
            state_best_checkpoint_id = resume_state.get('best_val_checkpoint_id')
            try:
                if not state_best_checkpoint:
                    raise ValueError('state best checkpoint path is missing')
                self._validate_best_checkpoint(
                    state_best_checkpoint,
                    checkpoint_id=state_best_checkpoint_id,
                    expected_psnr=state_best if state_best_checkpoint_id is not None else None,
                    expected_iter=state_best_iter if state_best_checkpoint_id is not None else None)
            except Exception as exc:
                get_root_logger().warning(f'Ignore invalid best record from training state: {exc}')
            else:
                self.best_val_psnr = state_best
                self.best_val_psnr_iter = state_best_iter
                self.best_val_checkpoint = state_best_checkpoint
                self.best_val_checkpoint_id = state_best_checkpoint_id
        if self.best_val_psnr_iter >= 0:
            self._save_best_metric_record()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        net_g_reg_ratio = 1
        normal_params = []
        normal_params = list(self.ddpm.parameters())
        optim_params_g = normal_params
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0.9**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

    def load_network(self, net, load_path, strict=False, param_key='params'):
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')

        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def load_bare_network(self, net, load_path, strict=True, param_key='params'):
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if "module." in k:
                load_net[k.replace("module.", "")] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'pad_left' in data:
            self.pad_left = data['pad_left'].to(self.device)
            self.pad_right = data['pad_right'].to(self.device)
            self.pad_top = data['pad_top'].to(self.device)
            self.pad_bottom = data['pad_bottom'].to(self.device)
    
    def norm_minus1_1(self, x):
        return (x - 0.5) * 2
    
    def norm_0_1(self, x):
        return (x + 1) * 0.5

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        pred_noise, noise, x_recon_out, supervised_l_list, supervised_r_list, l_MoE = self.ddpm(self.norm_minus1_1(self.gt), self.norm_minus1_1(self.lq),
                  train_type=self.opt['train'].get('train_type', None),
                  different_t_in_one_batch=self.opt['train'].get('different_t_in_one_batch', None),
                  clip_noise=self.opt['train'].get('clip_noise', None),
                  t_range=self.opt['train'].get('t_range', None),
                  frozen_denoise=self.opt['train'].get('frozen_denoise', None))

        supervised_l_list = [F.interpolate(output, size=self.gt.shape[2:], mode='bilinear', align_corners=False) for output in supervised_l_list]
        supervised_r_list = [F.interpolate(output, size=self.gt.shape[2:], mode='bilinear', align_corners=False) for output in supervised_r_list]

        r_gt, l_gt = self.decom_net(self.gt)

        x_recon_out = self.norm_0_1(x_recon_out)

        l_total = 0
        loss_dict = OrderedDict()

        l_diff = F.l1_loss(pred_noise, noise)
        l_l1 = F.l1_loss(x_recon_out, self.gt)
        l_retinex_l = [F.l1_loss(l, l_gt) for l in supervised_l_list]
        l_retinex_r = [F.l1_loss(r, r_gt) for r in supervised_r_list]
        l_retinex = 0.5 * sum(l_retinex_l) + sum(l_retinex_r)

        l_total += l_l1 + 0.2 * l_retinex + l_diff + l_MoE
        loss_dict['l_l1'] = l_l1
        loss_dict['l_diff'] = l_diff
        loss_dict['l_retinex'] = l_retinex
        loss_dict['l_MoE'] = l_MoE
        loss_dict['l_total'] = l_total
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def color_loss(self, lq, gt):
        lq_color_norm = torch.nn.functional.normalize(lq, p=2, dim=3)
        gt_color_norm = torch.nn.functional.normalize(gt, p=2, dim=3)
        l_color = torch.mean(1 - torch.sum(lq_color_norm * gt_color_norm, dim=3, keepdim=True))
        return l_color
    
    def gradient_loss(self, lq, gt):
        lq_grad = torch.abs(self.gradient(lq))
        gt_grad = torch.abs(self.gradient(gt))
        l_grad = torch.mean(torch.abs(lq_grad - gt_grad))
        return l_grad
    
    def gradient(self, image):
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        image_gray = F.pad(image_gray, self.padding, mode='replicate')
        gradient_x = F.conv2d(image_gray, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image_gray, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)

    def test(self):
        with torch.no_grad():
            self.bare_ddpm_model.eval()
            self.diff_sr, self.supervised_l_list, self.supervised_r_list = self.bare_ddpm_model.ddim_sample(self.norm_minus1_1(self.lq),
                                                continous=self.opt['val'].get('ret_process', False), 
                                                ddim_timesteps=self.opt['val'].get('ddim_timesteps', 25),
                                                return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                clip_noise=self.opt['val'].get('clip_noise', False),
                                                return_all=self.opt['val'].get('ret_all', False))
            
            self.diff_sr = self.norm_0_1(self.diff_sr)

            self.bare_ddpm_model.train()
            if hasattr(self, 'pad_left') and not self.opt['val'].get('ret_process', False):
                self.diff_sr = pad_tensor_back(self.diff_sr, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.lq = pad_tensor_back(self.lq, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.gt = pad_tensor_back(self.gt, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        #if self.opt['rank'] == 0:
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, test=False):
        test = self.opt['val'].get('test_flag', False)
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        metric_data = dict()
        metric_data_pytorch = dict()
        pbar = tqdm(total=len(dataloader), unit='item')
        if self.opt['val'].get('split_log', False):
            self.split_results = {}
            self.split_results['LOL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            visuals = self.get_current_visuals()
            gt_img = tensor2img([visuals['gt']], min_max=(0, 1))
            lq_img = tensor2img([visuals['lq']], min_max=(0, 1))
            sr_img = tensor2img([visuals['diff_sr']], min_max=(0, 1))
            l_list = tensor2img([visuals['l_list']], min_max=(0, 1))
            r_img = tensor2img([visuals['r_list']], min_max=(0, 1))

            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img
            metric_data_pytorch['img'] = sr_img
            metric_data_pytorch['img2'] = gt_img

            path = val_data['lq_path'][0]
            if self.opt['rank'] == 0:
                if save_img:
                    # save imgs
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                    save_img_retinex_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_retinex.png')
                    if not self.opt['val'].get('save_result_only', False):
                        imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)
                        imwrite(np.concatenate([l_list, r_img], axis=1), save_img_retinex_path)
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], f'{img_name}.png')
                        imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'pytorch' in opt_['type']:
                        self.metric_results[name] += calculate_metric(metric_data_pytorch, opt_).item()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            # tentative for out of GPU memory
            del self.lq, self.gt
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            if test is not True:
                for metric, value in self.metric_results.items():
                    if metric == "psnr":
                        if value > self.best_val_psnr:
                            self._save_new_best(value, current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def validation_UHD(self, dataloader, current_iter, tb_logger, save_img=False):
        self.nondist_validation_UHD(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation_UHD(self, dataloader, current_iter, tb_logger, save_img, test=False):
        test = self.opt['val'].get('test_flag', False)
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        metric_data = dict()
        metric_data_pytorch = dict()
        pbar = tqdm(total=len(dataloader), unit='item')
        if self.opt['val'].get('split_log', False):
            self.split_results = {}
            self.split_results['LOL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test_UHD()

            visuals = self.get_current_visuals_UHD()
            gt_img = tensor2img([visuals['gt']], min_max=(0, 1))
            sr_img = tensor2img([visuals['diff_sr']], min_max=(0, 1))

            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img
            metric_data_pytorch['img'] = sr_img
            metric_data_pytorch['img2'] = gt_img

            path = val_data['lq_path'][0]
            if self.opt['rank'] == 0:
                if save_img:
                    # save imgs
                    save_img_path = osp.join(self.opt['path']['visualization'], f'{img_name}.png')
                    imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'pytorch' in opt_['type']:
                        self.metric_results[name] += calculate_metric(metric_data_pytorch, opt_).item()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            # tentative for out of GPU memory
            del self.lq, self.gt
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            if test is not True:
                for metric, value in self.metric_results.items():
                    if metric == "psnr":
                        if value > self.best_val_psnr:
                            self._save_new_best(value, current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def pad_UHD(self, img, window_size=16):
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return img, mod_pad_h, mod_pad_w

    def pad_back_UHD(self, img, mod_pad_h, mod_pad_w):
        _, _, h, w = img.size()
        img = img[:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]
        return img

    def test_UHD(self):
        with torch.no_grad():
            self.bare_ddpm_model.eval()
            b, c, h, w = self.lq.shape
            if h < 1000 and w < 1000:
                self.diff_sr, _, _ = self.bare_ddpm_model.ddim_sample(self.norm_minus1_1(self.lq),
                                                                      continous=self.opt['val'].get('ret_process',
                                                                                                    False),
                                                                      ddim_timesteps=self.opt['val'].get(
                                                                          'ddim_timesteps', 25),
                                                                      return_pred_noise=self.opt['val'].get(
                                                                          'return_pred_noise', False),
                                                                      return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                                         False),
                                                                      ddim_discr_method=self.opt['val'].get(
                                                                          'ddim_discr_method', 'uniform'),
                                                                      ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                                      pred_type=self.opt['val'].get('pred_type',
                                                                                                    'noise'),
                                                                      clip_noise=self.opt['val'].get('clip_noise',
                                                                                                     False),
                                                                      return_all=self.opt['val'].get('ret_all', False))
            else:
                img_1 = self.lq[:, :, 0::2, 0::2]
                img_2 = self.lq[:, :, 0::2, 1::2]
                img_3 = self.lq[:, :, 1::2, 0::2]
                img_4 = self.lq[:, :, 1::2, 1::2]
                img_1, pad_h_1, pad_w_1 = self.pad_UHD(img_1)
                img_2, pad_h_2, pad_w_2 = self.pad_UHD(img_2)
                img_3, pad_h_3, pad_w_3 = self.pad_UHD(img_3)
                img_4, pad_h_4, pad_w_4 = self.pad_UHD(img_4)
                diff_sr_1, _, _ = self.bare_ddpm_model.ddim_sample(self.norm_minus1_1(img_1),
                                                                   continous=self.opt['val'].get('ret_process', False),
                                                                   ddim_timesteps=self.opt['val'].get('ddim_timesteps',
                                                                                                      25),
                                                                   return_pred_noise=self.opt['val'].get(
                                                                       'return_pred_noise', False),
                                                                   return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                                      False),
                                                                   ddim_discr_method=self.opt['val'].get(
                                                                       'ddim_discr_method', 'uniform'),
                                                                   ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                                   pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                                   clip_noise=self.opt['val'].get('clip_noise', False),
                                                                   return_all=self.opt['val'].get('ret_all', False))

                diff_sr_2, _, _ = self.bare_ddpm_model.ddim_sample(self.norm_minus1_1(img_2),
                                                                   continous=self.opt['val'].get('ret_process', False),
                                                                   ddim_timesteps=self.opt['val'].get('ddim_timesteps',
                                                                                                      25),
                                                                   return_pred_noise=self.opt['val'].get(
                                                                       'return_pred_noise', False),
                                                                   return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                                      False),
                                                                   ddim_discr_method=self.opt['val'].get(
                                                                       'ddim_discr_method', 'uniform'),
                                                                   ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                                   pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                                   clip_noise=self.opt['val'].get('clip_noise', False),
                                                                   return_all=self.opt['val'].get('ret_all', False))

                diff_sr_3, _, _ = self.bare_ddpm_model.ddim_sample(self.norm_minus1_1(img_3),
                                                                   continous=self.opt['val'].get('ret_process', False),
                                                                   ddim_timesteps=self.opt['val'].get('ddim_timesteps',
                                                                                                      25),
                                                                   return_pred_noise=self.opt['val'].get(
                                                                       'return_pred_noise', False),
                                                                   return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                                      False),
                                                                   ddim_discr_method=self.opt['val'].get(
                                                                       'ddim_discr_method', 'uniform'),
                                                                   ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                                   pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                                   clip_noise=self.opt['val'].get('clip_noise', False),
                                                                   return_all=self.opt['val'].get('ret_all', False))

                diff_sr_4, _, _ = self.bare_ddpm_model.ddim_sample(self.norm_minus1_1(img_4),
                                                                   continous=self.opt['val'].get('ret_process', False),
                                                                   ddim_timesteps=self.opt['val'].get('ddim_timesteps',
                                                                                                      25),
                                                                   return_pred_noise=self.opt['val'].get(
                                                                       'return_pred_noise', False),
                                                                   return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                                      False),
                                                                   ddim_discr_method=self.opt['val'].get(
                                                                       'ddim_discr_method', 'uniform'),
                                                                   ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                                   pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                                   clip_noise=self.opt['val'].get('clip_noise', False),
                                                                   return_all=self.opt['val'].get('ret_all', False))

                diff_sr_1 = self.pad_back_UHD(diff_sr_1, pad_h_1, pad_w_1)
                diff_sr_2 = self.pad_back_UHD(diff_sr_2, pad_h_2, pad_w_2)
                diff_sr_3 = self.pad_back_UHD(diff_sr_3, pad_h_3, pad_w_3)
                diff_sr_4 = self.pad_back_UHD(diff_sr_4, pad_h_4, pad_w_4)

                restored = torch.zeros_like(self.gt)
                restored[:, :, 0::2, 0::2] = diff_sr_1
                restored[:, :, 0::2, 1::2] = diff_sr_2
                restored[:, :, 1::2, 0::2] = diff_sr_3
                restored[:, :, 1::2, 1::2] = diff_sr_4
                self.diff_sr = restored

            self.diff_sr = self.norm_0_1(self.diff_sr)

            self.bare_ddpm_model.train()
            if hasattr(self, 'pad_left') and not self.opt['val'].get('ret_process', False):
                self.diff_sr = pad_tensor_back(self.diff_sr, self.pad_left, self.pad_right, self.pad_top,
                                               self.pad_bottom)
                self.lq = pad_tensor_back(self.lq, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.gt = pad_tensor_back(self.gt, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        logger = get_root_logger()
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)
        
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['lq'] = self.lq[:, :3, :, :].detach().cpu()
        out_dict['diff_sr'] = self.diff_sr.detach().cpu()
        resized_l = [torch.nn.functional.interpolate(torch.cat((l.detach().cpu(), l.detach().cpu(), l.detach().cpu()), dim=1), size=self.gt.shape[2:], mode='bilinear', align_corners=False) for l in self.supervised_l_list]
        l_list = torch.cat(resized_l, dim=3)
        resized_r = [torch.nn.functional.interpolate(r.detach().cpu(), size=self.gt.shape[2:], mode='bilinear', align_corners=False) for r in self.supervised_r_list]
        r_list = torch.cat(resized_r, dim=3)
        out_dict['l_list']= l_list.detach().cpu()
        out_dict['r_list']= r_list.detach().cpu()
        return out_dict

    def get_current_visuals_UHD(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['diff_sr'] = self.diff_sr.detach().cpu()
        return out_dict
    
    def save(self, epoch, current_iter, name=None):
        checkpoint_id = uuid.uuid4().hex
        if name is None:
            self.save_network(
                [self.ddpm],
                'net_g_ddpm',
                current_iter,
                param_key=['params'],
                checkpoint_id=checkpoint_id,
                checkpoint_metadata={'iter': current_iter, 'epoch': epoch})
            self.save_training_state(epoch, current_iter, checkpoint_id=checkpoint_id)
        else:
            self.save_network(
                [self.ddpm], 'net_g_best_ddpm_val', current_iter, param_key=['params'], checkpoint_id=checkpoint_id)

    @master_only
    def save_network(
            self,
            net,
            net_label,
            current_iter,
            param_key='params',
            checkpoint_id=None,
            checkpoint_metadata=None):
        """可靠保存网络；失败时抛错，禁止训练带病继续。"""
        if current_iter == -1:
            current_iter = 'latest'
        save_path = osp.join(self.opt['path']['models'], f'{net_label}_{current_iter}.pth')

        networks = net if isinstance(net, list) else [net]
        param_keys = param_key if isinstance(param_key, list) else [param_key]
        if len(networks) != len(param_keys):
            raise ValueError('The lengths of networks and param keys must match.')

        save_dict = {}
        for network, key_name in zip(networks, param_keys):
            bare_network = self.get_bare_model(network)
            state_dict = OrderedDict()
            for key, parameter in bare_network.state_dict().items():
                normalized_key = key[7:] if key.startswith('module.') else key
                state_dict[normalized_key] = parameter.detach().cpu()
            save_dict[key_name] = state_dict
        if checkpoint_id is not None:
            save_dict['checkpoint_id'] = checkpoint_id
        if checkpoint_metadata is not None:
            save_dict['metadata'] = checkpoint_metadata

        _atomic_torch_save(save_dict, save_path, required_keys=param_keys)

    @master_only
    def save_training_state(self, epoch, current_iter, checkpoint_id=None):
        """可靠保存恢复状态，并把跨重启最佳指标写入 state。"""
        if current_iter == -1:
            return
        if not checkpoint_id:
            raise ValueError('checkpoint_id is required for a resumable training state.')

        state = {
            'epoch': epoch,
            'iter': current_iter,
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'schedulers': [scheduler.state_dict() for scheduler in self.schedulers],
            'best_val_psnr': float(self.best_val_psnr),
            'best_val_psnr_iter': int(self.best_val_psnr_iter),
            'best_val_checkpoint': self.best_val_checkpoint,
            'best_val_checkpoint_id': self.best_val_checkpoint_id,
            'checkpoint_id': checkpoint_id
        }
        save_path = osp.join(self.opt['path']['training_states'], f'{current_iter}.state')
        _atomic_torch_save(
            state,
            save_path,
            required_keys=(
                'epoch', 'iter', 'optimizers', 'schedulers', 'best_val_psnr', 'best_val_psnr_iter',
                'best_val_checkpoint', 'best_val_checkpoint_id', 'checkpoint_id'))
