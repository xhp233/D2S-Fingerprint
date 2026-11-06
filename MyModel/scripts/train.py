import os
import sys
import json
import torch
import numpy as np
import torch.utils.data as data_cls
import torch.nn.functional as F
import random
import math
import re
import argparse
import gc
import ctypes
from datetime import datetime

import sys


from MyModel.utils.train_manager import TrainingControlManager, SimplePhaseManager, build_optimizer, assert_resume_consistency
from MyModel.utils.ckpt_manager import CheckpointManager
from MyModel.utils.utils import collect_all_files, split_files_cfpose, split_files_fixed_uwa, split_files_mixed_protocol, get_finger_id, collate_list, _get_unsw_subject
from MyModel.utils.utils import count_model_parameters, load_checkpoint_into_model
from MyModel.utils.utils import compute_pairwise_similarity, move_sample_to_device
from MyModel.model import FingerprintNet
from MyModel.datasets import FingerprintDataset, BalancedBatchSampler
from MyModel.scripts.test import test_benchmark





def batch_hard_triplet_loss_phase_2_3(dist_mat, labels, margin=0.2, hard_neg_quantile=0.0):
    B = dist_mat.size(0)
    labels = labels.unsqueeze(0)  # [1, B]
    mask_pos = (labels == labels.transpose(0, 1)).float()  # [B, B]
    mask_pos = mask_pos * (1.0 - torch.eye(B, device=dist_mat.device))
    mask_neg = (labels != labels.transpose(0, 1)).float()  # [B, B]

    hard_pos, _ = torch.max(dist_mat * mask_pos, dim=1)

    dist_mat_neg = dist_mat * mask_neg + (1.0 - mask_neg) * 1e9
    valid_neg_count = mask_neg.sum(dim=1).long()

    if hard_neg_quantile <= 0.0:
        hard_neg, _ = torch.min(dist_mat_neg, dim=1)
    else:
        sorted_neg, _ = torch.sort(dist_mat_neg, dim=1)
        q = min(max(float(hard_neg_quantile), 0.0), 1.0)
        q_idx = ((valid_neg_count.float() - 1.0) * q).long().clamp(min=0)
        hard_neg = sorted_neg.gather(1, q_idx.unsqueeze(1)).squeeze(1)

    losses = F.softplus(hard_pos - hard_neg + margin)

    valid_mask = (mask_pos.sum(dim=1) > 0).float()
    if valid_mask.sum() > 0:
        return (losses * valid_mask).sum() / valid_mask.sum()
    else:
        return losses.mean() * 0.0


def compute_triplet_loss_by_phase(dist_mat, labels, curr_phase, curr_margin, w_all, w_hard, hard_neg_quantile):
    if labels.numel() <= 1:
        return dist_mat.mean() * 0.0

    if curr_phase <= 1:
        return batch_all_triplet_loss_phase_1_2(dist_mat, labels, margin=curr_margin)
    if curr_phase == 2:
        l_all = batch_all_triplet_loss_phase_1_2(dist_mat, labels, margin=curr_margin)
        l_hard = batch_hard_triplet_loss_phase_2_3(
            dist_mat,
            labels,
            margin=curr_margin,
            hard_neg_quantile=float(hard_neg_quantile),
        )
        return w_all * l_all + w_hard * l_hard
    return batch_hard_triplet_loss_phase_2_3(
        dist_mat,
        labels,
        margin=curr_margin,
        hard_neg_quantile=float(hard_neg_quantile),
    )


def compute_batch_hard_stats_for_ntn(dist_mat, labels, hard_neg_quantile=0.0):
    B = dist_mat.size(0)
    L = labels.unsqueeze(0)
    mask_pos = (L == L.transpose(0, 1)).float()
    mask_neg = (L != L.transpose(0, 1)).float()
    mask_pos = mask_pos * (1.0 - torch.eye(B, device=dist_mat.device))

    hard_pos, _ = torch.max(dist_mat * mask_pos, dim=1)
    dist_mat_neg = dist_mat * mask_neg + (1.0 - mask_neg) * 1e9
    valid_neg_count = mask_neg.sum(dim=1).long()

    if hard_neg_quantile <= 0.0:
        hard_neg, _ = torch.min(dist_mat_neg, dim=1)
    else:
        sorted_neg, _ = torch.sort(dist_mat_neg, dim=1)
        q = min(max(float(hard_neg_quantile), 0.0), 1.0)
        q_idx = ((valid_neg_count.float() - 1.0) * q).long().clamp(min=0)
        hard_neg = sorted_neg.gather(1, q_idx.unsqueeze(1)).squeeze(1)

    valid_anchor_mask = (mask_pos.sum(dim=1) > 0)
    return hard_pos, hard_neg, valid_anchor_mask


def pair_tail_alignment_loss(sim_matrix, labels, tail_threshold=0.95, tail_alpha=4.0, tail_tau=0.05):
    B = sim_matrix.size(0)
    if B <= 1:
        return sim_matrix.mean() * 0.0

    labels_col = labels.unsqueeze(1)
    labels_row = labels.unsqueeze(0)
    pos_mask = (labels_col == labels_row)
    neg_mask = ~pos_mask
    eye = torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
    pos_mask = pos_mask & (~eye)
    neg_mask = neg_mask & (~eye)

    s = sim_matrix.clamp(min=1e-5, max=1.0 - 1e-5)
    y = pos_mask.float()
    base_bce = -(y * torch.log(s) + (1.0 - y) * torch.log(1.0 - s))

    tail_tau = max(float(tail_tau), 1e-4)
    tail_threshold = float(tail_threshold)
    tail_alpha = max(float(tail_alpha), 0.0)
    neg_tail_boost = 1.0 + tail_alpha * torch.sigmoid((s - tail_threshold) / tail_tau)
    weights = torch.ones_like(base_bce)
    weights = torch.where(neg_mask, neg_tail_boost, weights)

    valid_mask = (~eye)
    loss = (base_bce * weights)[valid_mask]
    if loss.numel() == 0:
        return sim_matrix.mean() * 0.0
    return loss.mean()


def batch_all_triplet_loss_phase_1_2(dist_mat, labels, margin=0.2):
    # 2. 构建三元组掩码 [B, B, B]
    B = dist_mat.size(0)
    labels = labels.unsqueeze(0)  # [1, B]
    
    # is_pos[i, j] = 1 if labels[i] == labels[j]
    is_pos = (labels == labels.transpose(0, 1))
    # is_neg[i, k] = 1 if labels[i] != labels[k]
    is_neg = (labels != labels.transpose(0, 1))
    
    # 扩展维度以广播
    dist_ap = dist_mat.unsqueeze(2)
    dist_an = dist_mat.unsqueeze(1)
    
    # 3. 计算所有可能的 Triplet Loss
    triplet_loss = F.softplus(dist_ap - dist_an + margin)
    
    # 4. 应用掩码
    mask = is_pos.unsqueeze(2) & is_neg.unsqueeze(1)
    eye = torch.eye(B, device=dist_mat.device).bool()
    mask = mask & (~eye.unsqueeze(2))
    
    # 5. 计算平均 Loss
    valid_loss = triplet_loss[mask]
    
    if valid_loss.numel() == 0:
        return torch.tensor(0.0, device=dist_mat.device, requires_grad=True)
    
    return valid_loss.mean()


def train_main():

    from MyModel.utils.cli import data_paths, parse_with_config

    # ==========================================================================
    # 命令行参数配置
    # ==========================================================================
    parser = argparse.ArgumentParser(description='指纹识别模型训练与可视化工具')
    
    # --------------------------------------------------------------------------
    # 数据集相关参数
    # --------------------------------------------------------------------------
    parser.add_argument('--data_name', type=str, default='unsw', choices=['cfpose', 'unsw', 'mixed'],
                       help='数据集名称（控制数据路径和文件名解析规则），mixed 表示混合 cfpose 和 unsw')
    parser.add_argument('--mixed_unsw_sampling_ratio', type=float, default=0.7,
                       help='mixed 训练时每个 batch 中 UNSW 身份占比（0~1，默认 0.7）')
    parser.add_argument('--cfpose_test_count', type=int, default=1200,
                       help='CFPose 划分中测试集样本数（按有序文件前缀切分）')
    parser.add_argument('--cfpose_val_count', type=int, default=20,
                       help='CFPose 划分中验证集样本数（紧随测试集）')
    parser.add_argument('--cfpose_train_count', type=int, default=180,
                       help='CFPose 划分中训练集样本数（其余部分）')
    
    # --------------------------------------------------------------------------
    # 评估策略配置
    # --------------------------------------------------------------------------
    parser.add_argument('--val_every_n_epochs', type=int, default=5,
                       help='每 N 个 epoch 执行一次验证集评估')
    parser.add_argument('--benchmark_batch_size', type=int, default=32,
                       help='benchmark 评估的 batch size')
    
    # --------------------------------------------------------------------------
    # 训练模式与流程参数
    # --------------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=200,
                       help='总训练轮数')
    
    # --------------------------------------------------------------------------
    # Batch 与采样参数
    # --------------------------------------------------------------------------
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载的 worker 数量')
    
    # --------------------------------------------------------------------------
    # 优化器与损失函数参数
    # --------------------------------------------------------------------------
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减（L2 正则化）')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Triplet Loss 的边界值 margin')
    parser.add_argument('--warm_up_epochs', type=int, default=30,
                       help='损失权重平滑过渡的 warm-up 轮数')
    parser.add_argument('--batch_all_epochs', type=int, default=30,
                       help='使用 batch-all triplet loss 的轮数（之后切换到 batch-hard）')
    parser.add_argument('--mixed_hard_epochs', type=int, default=30,
                       help='混合 batch-all 和 batch-hard 并使用渐进 margin 的轮数')
    parser.add_argument('--hard_neg_quantile', type=float, default=0.0,
                       help='Batch-Hard 负样本距离分位点（0=最难负样本，建议 0.1~0.3 降低极端噪声影响）')
    parser.add_argument('--ntn_reg_lambda', type=float, default=1e-3,
                       help='NTN/fusion 参数的 L2 正则系数')
    parser.add_argument('--ntn_temperature', type=float, default=1.0,
                       help='NTN 分支温度缩放（>1 可减缓 Sigmoid 饱和）')
    parser.add_argument('--l2_temperature', type=float, default=1.0,
                       help='L2 分支温度缩放（>1 可放缓 exp(-d) 的贴顶速度）')
    parser.add_argument('--scorer_weight_decay_scale', type=float, default=10.0,
                       help='Scorer/fusion 参数组的权重衰减倍率（相对 weight_decay）')
    parser.add_argument('--enable_pair_loss', action='store_true',
                       help='启用 pair-tail 辅助损失')
    parser.add_argument('--pair_loss_weight', type=float, default=0.2,
                       help='双目标中 pair-tail 辅助损失权重')
    parser.add_argument('--pair_tail_threshold', type=float, default=0.95,
                       help='pair-tail 中判定高分 impostor 的阈值')
    parser.add_argument('--pair_tail_alpha', type=float, default=4.0,
                       help='pair-tail 对高分 impostor 的加权强度')
    parser.add_argument('--pair_tail_tau', type=float, default=0.05,
                       help='pair-tail 的平滑参数')
    parser.add_argument('--enable_arcface', action='store_true',
                       help='启用 ArcFace 分类损失')
    parser.add_argument('--arcface_s', type=float, default=32.0,
                       help='ArcFace 的缩放因子 s（指纹建议 32.0）')
    parser.add_argument('--arcface_m', type=float, default=0.3,
                       help='ArcFace 的角度裕度 m（微调建议从 0.3 开始）')
    parser.add_argument('--arcface_weight', type=float, default=1.0,
                       help='ArcFace 损失权重')
    parser.add_argument('--enable_dynamic_pair_tail_threshold', action='store_true',
                       help='启用动态 pair-tail 阈值：按本 epoch impostor 分布分位点更新，下 epoch 生效')
    parser.add_argument('--dynamic_pair_tail_quantile', type=float, default=0.9,
                       help='动态 pair-tail 阈值使用的 impostor 分位点（默认 0.9）')
    parser.add_argument('--dynamic_pair_tail_min', type=float, default=0.2,
                       help='动态 pair-tail 阈值下界')
    parser.add_argument('--dynamic_pair_tail_max', type=float, default=0.99,
                       help='动态 pair-tail 阈值上界')
    parser.add_argument('--dynamic_pair_tail_momentum', type=float, default=0.0,
                       help='动态 pair-tail 阈值 EMA 动量（0 表示直接采用新分位值）')
    parser.add_argument('--dynamic_pair_tail_max_samples', type=int, default=200000,
                       help='每个 epoch 统计 impostor 分数的最大采样数（防止内存增长）')
    parser.add_argument('--enable_dynamic_pair_weight', action='store_true',
                       help='启用动态 pair 权重：Phase0-2 固定，Phase3(Batch-Hard) 使用可学习权重')
    parser.add_argument('--dynamic_pair_weight_pre_hard', type=float, default=0.2,
                       help='动态 pair 权重启用时，Phase0-2 使用的固定权重')
    parser.add_argument('--dynamic_pair_weight_init', type=float, default=0.2,
                       help='动态 pair 权重启用时，Phase3 可学习权重的初始化值')
    parser.add_argument('--dynamic_pair_weight_max', type=float, default=0.4,
                       help='动态 pair 权重启用时，Phase3 可学习权重上限')
    parser.add_argument('--dynamic_pair_weight_lr_scale', type=float, default=0.1,
                       help='动态 pair 权重参数学习率倍率（相对主 lr）')
    parser.add_argument('--enable_feature_constancy', action='store_true',
                       help='启用特征一致性正则：约束当前模型特征与教师模型特征接近')
    parser.add_argument('--feature_constancy_weight', type=float, default=0.01,
                       help='特征一致性损失权重')
    parser.add_argument('--feature_constancy_ckpt', type=str, default='',
                       help='教师模型 checkpoint 路径（为空时回退到 load_path）')
    parser.add_argument('--enable_mixed_dynamic_domain_weight', action='store_true',
                       help='仅 mixed 模式启用：UNSW/CFPose Triplet Loss 动态不确定性加权')
    parser.add_argument('--mixed_domain_weight_init_log_var', type=float, default=0.0,
                       help='mixed 动态域权重的 log_var 初始值（UNSW/CFPose 共用）')
    parser.add_argument('--mixed_domain_weight_lr_scale', type=float, default=0.1,
                       help='mixed 动态域权重参数学习率倍率（相对主 lr）')
    parser.add_argument('--mixed_static_weight_unsw', type=float, default=1.0,
                       help='mixed 模式下 UNSW 子域 triplet loss 的静态权重')
    parser.add_argument('--mixed_static_weight_cfpose', type=float, default=1.0,
                       help='mixed 模式下 CFPose 子域 triplet loss 的静态权重')
    
    # --------------------------------------------------------------------------
    # 模型架构参数
    # --------------------------------------------------------------------------
    parser.add_argument('--num_layers', type=int, default=3,
                       help='超图神经网络的层数')
    parser.add_argument('--knn_k', type=int, default=25,
                       help='KNN 构建超图时的邻居数量 k')
    parser.add_argument('--tensor_neurons', type=int, default=32,
                       help='Tensor Network 中隐藏层的神经元数量')
    
    # --------------------------------------------------------------------------
    # Batch 并行化配置
    # --------------------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=32,
                       help='训练的 batch size（默认为 32，根据显存调整）')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                       help='评估的 batch size（默认为 32，根据显存调整）')
    parser.add_argument('--no_pin_memory', action='store_false', dest='pin_memory',
                       help='禁用 pin_memory 加速数据传输')
    parser.set_defaults(pin_memory=True)
    parser.add_argument('--topk_pool', type=int, default=50,
                       help='Top-K 池化：每个图保留的节点数量')
    parser.add_argument('--fps_alpha', type=float, default=0.2,
                       help='加权 FPS 中距离的权重（多样性 vs 重要性）')
    parser.add_argument('--attn_dropout', type=float, default=0.5,
                       help='注意力分数的 dropout 概率（训练时）')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='通用 dropout 概率（用于 Encoder、Scorer、GNN 等模块）')
    parser.add_argument('--gnn_dropout', type=float, default=0.1,
                       help='GNN 输出层的 dropout 概率')
    parser.add_argument('--drop_edge_prob', type=float, default=0.1,
                       help='超图边的随机丢弃概率')
    parser.add_argument('--feature_jitter', type=float, default=0.05,
                       help='特征随机抖动的方差')
    parser.add_argument('--disable_topology_score', action='store_true',
                       help='消融开关：禁用拓扑分数融合（Readout 仅使用纹理分数）')
    parser.add_argument('--graph_backbone', type=str, default='hgnn', choices=['hgnn', 'gcn'],
                       help='图骨干类型：hgnn 或 gcn')
    parser.add_argument('--disable_contraction', action='store_true',
                       help='消融开关：保留 HGNN 高阶超图骨干，但关闭动态收缩（H 全程固定为 H^0/static-H）。'
                            '用于把"逐层收缩"从"高阶超图"中单独剥离出来。')
    parser.add_argument('--baseline_b', action='store_true',
                       help='启用 baseline_b 模式：纯纹理特征流，无 GNN 和 GF-NMS')
    parser.add_argument('--freeze_cnn', action='store_true',
                       help='冻结 CNN 骨干网络的所有参数（包括 BN 层）')
    parser.add_argument('--peft_metric_tuning', action='store_true',
                       help='仅训练 InteractionScorer 与 attn_head，冻结其余参数（小样本迁移）')
    parser.add_argument('--hgnn_train_last_n_layers', type=int, default=0,
                       help='冻结其余模块，仅训练 HGNN 最后 N 层 + attn_head + scorer/fusion_w（0=禁用）')
    
    # --------------------------------------------------------------------------
    # 训练设备与精度
    # --------------------------------------------------------------------------
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备（cuda 或 cpu）')
    parser.add_argument('--augment', action='store_true',
                       help='启用随机旋转和平移数据增强')
    parser.add_argument('--augment_profile', type=str, default='default', choices=['default', 'micro'],
                       help='增强配置：default 为历史增强，micro 为精细增强（小角度/小平移）')
    parser.add_argument('--micro_aug_max_rotate', type=float, default=5.0,
                       help='micro 增强下最大旋转角度（度）')
    parser.add_argument('--micro_aug_max_translate_ratio', type=float, default=0.02,
                       help='micro 增强下最大平移比例（相对图像宽高）')
    parser.add_argument('--enable_random_illum_gradient', action='store_true',
                       help='启用随机光照梯度增强（随机方向乘性光照场，支持非线性强度分布）')
    parser.add_argument('--illum_gradient_prob', type=float, default=0.65,
                       help='随机光照梯度增强触发概率（仅训练集）')
    parser.add_argument('--illum_gradient_strength_min', type=float, default=0.08,
                       help='随机光照梯度增强强度下界')
    parser.add_argument('--illum_gradient_strength_max', type=float, default=0.30,
                       help='随机光照梯度增强强度上界')
    parser.add_argument('--illum_gradient_gamma_min', type=float, default=0.7,
                       help='随机光照梯度非线性 gamma 下界')
    parser.add_argument('--illum_gradient_gamma_max', type=float, default=1.8,
                       help='随机光照梯度非线性 gamma 上界')
    parser.add_argument('--use_ntn', dest='use_ntn', action='store_true',
                       help='启用 NTN 打分器（默认启用）')
    
    # --------------------------------------------------------------------------
    # 检查点管理参数
    # --------------------------------------------------------------------------
    parser.add_argument('--save_path', type=str, default='',
                       help='可选：训练结束时保存最终检查点的完整路径')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts',
                       help='检查点目录（存储 last.pt, best.pt, manifest.json）')
    parser.add_argument('--save_best_k', type=int, default=10,
                       help='保留前 K 个最佳检查点（设为 0 禁用）')
    parser.add_argument('--ckpt_metric', type=str, default='eval_eer',
                       choices=['eval_loss', 'eval_acc', 'eval_eer', 'train_loss', 'train_acc'],
                       help='用于选择最佳检查点的指标')
    parser.add_argument('--ckpt_mode', type=str, default='min',
                       choices=['', 'min', 'max'],
                       help='覆盖指标的优化方向（默认根据指标推断：EER/Loss 为 min，Acc 为 max）')
    parser.add_argument('--load_path', type=str, default='',
                       help='要加载的预训练检查点路径（如：my_model_new.pt）')
    parser.add_argument('--load_weights_only', action='store_true',
                       help='仅加载模型权重进行迁移微调（不做严格续训一致性校验，不恢复优化器/epoch/phase状态）')
    parser.add_argument('--conv1_ft', action='store_true',
                       help='启用 conv1 微调模式：从检查点加载所有参数，仅进行 conv1 微调')
    parser.add_argument('--train_conv1', action='store_true',
                       help='启用从零开始训练 conv1 的模式：不使用自蒸馏，直接训练 conv1 层')
    parser.add_argument('--only_cfpose_test', action='store_true',
                       help='仅执行 CFPose 的测试，跳过 UNSW 的测试以节省时间')
    
    # --------------------------------------------------------------------------
    # 早停机制
    # --------------------------------------------------------------------------
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='早停耐心值：当val_eer在N次val内无改进时提前停止训练')
    
    # --------------------------------------------------------------------------
    # 随机种子
    # --------------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（确保可重复性）')
    
    # --------------------------------------------------------------------------
    # 内存管理
    # --------------------------------------------------------------------------
    parser.add_argument('--disable_malloc_trim', action='store_false', dest='enable_malloc_trim',
                       help='禁用评估阶段后的 malloc_trim 内存归还（默认启用）')
    parser.set_defaults(enable_malloc_trim=True)
    parser.add_argument('--disable_ram_usage_log', action='store_false', dest='log_ram_usage',
                       help='禁用 RAM RSS 监控日志写入 TensorBoard（默认启用）')
    parser.set_defaults(log_ram_usage=True)
    
    # --------------------------------------------------------------------------
    # TensorBoard 日志详细程度
    # --------------------------------------------------------------------------
    parser.add_argument('--tb_verbose', action='store_true',
                       help='启用详细的 TensorBoard 日志（包括直方图、分布等）')
    
    parser.add_argument('--data_root', default=os.environ.get('HGNN_DATA_ROOT', 'data'), help='Root containing cfpose/ and unsw/ dataset folders')
    parser.add_argument('--config', default=None, help='Explicit JSON configuration; command-line flags override it')
    parser.add_argument('--log_root', default='runs/tensorboard', help='TensorBoard output root')

    args = parse_with_config(parser)
    DATA_CONFIG = data_paths(args.data_root)
    ctrl = TrainingControlManager.from_args(args)

    # Redirect stdout to a log file in the ckpt directory.
    # Keep stderr on terminal so tqdm progress bars do not pollute log files.
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_file = os.path.join(args.ckpt_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    class Logger(object):
        def __init__(self, filename, stream):
            self.terminal = stream
            self.log = open(filename, "a", encoding='utf-8', buffering=1)
        def write(self, message):
            # 检查是否需要过滤
            if any(keyword in message for keyword in ['Step', 'loss=', 'l_triplet=', 'l_tail=', 'margin=', 'Logits=', 'Prob_mean=']):
                # 只在终端显示，不写入log文件
                self.terminal.write(message)
            else:
                # 保留的信息：参数打印、模型参数量统计、epoch信息、val/test结果
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger(log_file, sys.stdout)
    print(f'Logging to {log_file}')
    print('Logging mode: stdout -> file+terminal (params, model stats, val/test results), stderr -> terminal only (tqdm isolated).')
    print(f"Arguments: {json.dumps(vars(args), indent=2)}")
    print(f"Control manager: {json.dumps(ctrl.to_dict(), indent=2)}")
    
    # 初始化 TensorBoard writer
    run_name = f"{args.data_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join(args.log_root, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'TensorBoard logs: {log_dir}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f'Device: {device}')

    # checkpoint manager
    # EER should be minimized
    metric_default_mode = 'min'
        
    ckpt_mode = args.ckpt_mode if args.ckpt_mode in ('min', 'max') else metric_default_mode
    ckpt_mgr = CheckpointManager(
        ckpt_dir=args.ckpt_dir,
        keep_last=True,
        save_best_k=args.save_best_k,
        mode=ckpt_mode,
    )

    # 加载数据集文件
    train_files = []
    val_files = []
    test_files = []
    test_ids = []

    # build train/val/test split by id to avoid leakage
    all_files, cfpose_all_files, unsw_all_files = collect_all_files(args.data_name, DATA_CONFIG)

    if args.data_name == 'mixed':
        print(
            f"[Data] Mixed source loaded: total={len(all_files)} "
            f"(cfpose={len(cfpose_all_files)}, unsw={len(unsw_all_files)})"
        )

    if args.data_name == 'cfpose':
        print(
            "Using CFPose split protocol "
            f"(Test={args.cfpose_test_count}, Val={args.cfpose_val_count}, Train={args.cfpose_train_count})"
        )
        train_files, val_files, test_files, test_ids = split_files_cfpose(
            all_files,
            test_count=args.cfpose_test_count,
            val_count=args.cfpose_val_count,
            train_count=args.cfpose_train_count,
        )
        # 将 val 数据合并到 train 中
        train_files.extend(val_files)
        val_files = []  # 清空 val 数据
        print(f"[Data] Merged val into train: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    elif args.data_name == 'unsw':
        print("Using fixed UWA split protocol (First 5970 Test, Middle 101-110 Val, 111-150 Train)")
        train_files, val_files, test_files, test_ids = split_files_fixed_uwa(all_files)
        # 将 val 数据合并到 train 中
        train_files.extend(val_files)
        val_files = []  # 清空 val 数据
        print(f"[Data] Merged val into train: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    elif args.data_name == 'mixed':
        train_files, val_files, test_files, test_ids = split_files_mixed_protocol(
            cfpose_all_files,
            unsw_all_files,
            seed=args.seed,
            cfpose_test_count=args.cfpose_test_count,
            cfpose_val_count=args.cfpose_val_count,
            cfpose_train_count=args.cfpose_train_count,
        )
        # 将 val 数据合并到 train 中
        train_files.extend(val_files)
        val_files = []  # 清空 val 数据
        print(f"[Data] Merged val into train: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    else:
        raise ValueError(f"Test protocol not defined for data_name '{args.data_name}'.")

    print(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # 保存原始的命令行参数（sh 中定义的超参数）
    original_args = vars(args).copy()
    
    # Load checkpoint if provided
    loaded_ckpt = None
    start_epoch = 0  # 初始化 start_epoch
    global_step = 0  # 初始化 global_step
    
    if args.load_path:
        # 临时创建模型用于加载 checkpoint
        temp_model = FingerprintNet(
            feat_dim=128,
            num_layers=ctrl.num_layers,
            knn_k=ctrl.knn_k,
            topk_pool=ctrl.topk_pool,
            attn_dropout=ctrl.attn_dropout,
            fps_alpha=ctrl.fps_alpha,
            tensor_neurons=ctrl.tensor_neurons,
            dropout_rate=ctrl.dropout_rate,
            gnn_dropout=ctrl.gnn_dropout,
            drop_edge_prob=ctrl.drop_edge_prob,
            feature_jitter=ctrl.feature_jitter,
            ntn_temperature=ctrl.ntn_temperature,
            l2_temperature=ctrl.l2_temperature,
            use_ntn=ctrl.use_ntn,
            disable_topology_score=ctrl.disable_topology_score,
            graph_backbone=ctrl.graph_backbone,
        ).to(device)
        
        loaded_ckpt = load_checkpoint_into_model(temp_model, args.load_path, device=device)
        if args.load_weights_only:
            print('[Resumer] load_weights_only=True -> skip strict resume consistency check.')
        elif loaded_ckpt and 'args' in loaded_ckpt:
            # 如果启用了 conv1_ft，从检查点读取所有参数，然后用命令行参数覆盖
            if args.conv1_ft:
                print('[Conv1FT] Loading parameters from checkpoint, then overriding with command-line arguments...')
                # 保存 conv1_ft 标志
                conv1_ft_flag = args.conv1_ft
                # 从检查点加载 args
                ckpt_args = loaded_ckpt.get('args', {})
                # 先将所有参数设置为检查点中的值
                for key, value in ckpt_args.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                # 然后用命令行参数覆盖
                for key, value in original_args.items():
                    if key != 'load_path':  # 保持 load_path 不变
                        setattr(args, key, value)
                # 恢复 conv1_ft 标志
                args.conv1_ft = conv1_ft_flag
                # 重新构建 ctrl
                ctrl = TrainingControlManager.from_args(args)
                # 在 conv1_ft 模式下，不恢复 epoch 和 global_step，从 0 开始
                start_epoch = 0
                global_step = 0
                print('[Conv1FT] Parameters initialized from checkpoint, overridden with command-line arguments. Starting from epoch 1.')
            else:
                assert_resume_consistency(args, loaded_ckpt.get('args', {}))
                print('[Resumer] Strict consistency check passed.')
        else:
            raise RuntimeError('[Resumer] Checkpoint missing args; strict resume consistency requires args in checkpoint.')
    
    print(f'[Conv1FT] Using epochs value: {args.epochs}')

    # 创建最终的模型，使用更新后的 ctrl
    model = FingerprintNet(
        feat_dim=128,
        num_layers=ctrl.num_layers,
        knn_k=ctrl.knn_k,
        topk_pool=ctrl.topk_pool,
        attn_dropout=ctrl.attn_dropout,
        fps_alpha=ctrl.fps_alpha,
        tensor_neurons=ctrl.tensor_neurons,
        dropout_rate=ctrl.dropout_rate,
        gnn_dropout=ctrl.gnn_dropout,
        drop_edge_prob=ctrl.drop_edge_prob,
        feature_jitter=ctrl.feature_jitter,
        ntn_temperature=ctrl.ntn_temperature,
        l2_temperature=ctrl.l2_temperature,
        use_ntn=ctrl.use_ntn,
        disable_topology_score=ctrl.disable_topology_score,
        graph_backbone=ctrl.graph_backbone,
        disable_contraction=ctrl.disable_contraction,
    ).to(device)

    # 如果有 checkpoint，重新加载到新模型
    if args.load_path and loaded_ckpt:
        load_checkpoint_into_model(model, args.load_path, device=device)

    # 统计模型参数量
    count_model_parameters(model, verbose=True)

    def _enforce_frozen_modules_eval(target_model):
        if (args.freeze_cnn or args.peft_metric_tuning or args.hgnn_train_last_n_layers > 0) and hasattr(target_model, 'encoder'):
            # Keep frozen CNN in eval mode so BN running stats are not updated.
            target_model.encoder.eval()
        if args.peft_metric_tuning and hasattr(target_model, 'gnn'):
            target_model.gnn.eval()

    def _extract_hgnn_stage_index(param_name: str):
        patterns = [
            r'GNN_layer_(\d+)',
            r'NonLocal_(\d+)',
            r'update_graph_(\d+)',
            r'layer(\d+)',
        ]
        for pat in patterns:
            m = re.search(pat, param_name)
            if m is not None:
                return int(m.group(1))
        return None

    def _set_module_trainable(module, is_trainable: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = bool(is_trainable)

    def _apply_trainable_policy(target_model):
        if args.peft_metric_tuning:
            if args.freeze_cnn:
                print('[TrainPolicy] peft_metric_tuning 与 freeze_cnn 同时启用，按 peft_metric_tuning 执行。')
            if args.hgnn_train_last_n_layers > 0:
                print('[TrainPolicy] peft_metric_tuning 与 hgnn_train_last_n_layers 同时启用，按 peft_metric_tuning 执行。')
            if not hasattr(target_model, 'scorer'):
                raise RuntimeError('peft_metric_tuning requires model.scorer, please enable --use_ntn.')

            for param in target_model.parameters():
                param.requires_grad = False

            _set_module_trainable(getattr(target_model, 'attn_head', None), True)
            _set_module_trainable(getattr(target_model, 'scorer', None), True)
            _enforce_frozen_modules_eval(target_model)

            total_params = sum(1 for _ in target_model.parameters())
            trainable_params = sum(int(param.requires_grad) for param in target_model.parameters())
            print(
                '=== PEFT 策略：仅训练 InteractionScorer 与 attn_head，冻结其余模块 === '
                f'({trainable_params}/{total_params} trainable tensors)'
            )
            return

        if args.hgnn_train_last_n_layers > 0:
            if not hasattr(target_model, 'gnn'):
                raise RuntimeError('hgnn_train_last_n_layers requires model.gnn.')

            for param in target_model.parameters():
                param.requires_grad = False

            stage_indices = sorted(
                {
                    idx
                    for name, _ in target_model.gnn.named_parameters()
                    for idx in [ _extract_hgnn_stage_index(name) ]
                    if idx is not None
                }
            )
            if len(stage_indices) == 0:
                raise RuntimeError('Could not infer HGNN stage indices from parameter names.')

            keep_n = min(int(args.hgnn_train_last_n_layers), len(stage_indices))
            keep_stages = set(stage_indices[-keep_n:])

            gnn_total_tensors = 0
            gnn_trainable_tensors = 0
            for name, param in target_model.gnn.named_parameters():
                gnn_total_tensors += 1
                idx = _extract_hgnn_stage_index(name)
                if idx is not None and idx in keep_stages:
                    param.requires_grad = True
                    gnn_trainable_tensors += 1

            _set_module_trainable(getattr(target_model, 'attn_head', None), True)
            if hasattr(target_model, 'scorer'):
                _set_module_trainable(getattr(target_model, 'scorer', None), True)
            if hasattr(target_model, 'fusion_w'):
                target_model.fusion_w.requires_grad = True

            _enforce_frozen_modules_eval(target_model)
            total_params = sum(1 for _ in target_model.parameters())
            trainable_params = sum(int(param.requires_grad) for param in target_model.parameters())
            print(
                '=== Stage-wise 策略：训练 HGNN 最后 N 层 + attn_head + scorer/fusion_w === '
                f'(N={int(args.hgnn_train_last_n_layers)}, keep_stages={sorted(list(keep_stages))}, '
                f'gnn_trainable_tensors={gnn_trainable_tensors}/{gnn_total_tensors}, '
                f'{trainable_params}/{total_params} trainable tensors)'
            )
            return

        if args.freeze_cnn:
            print('=== 冻结 CNN 骨干网络所有参数（含 BN 仿射参数）===')
            _set_module_trainable(getattr(target_model, 'encoder', None), False)
            _enforce_frozen_modules_eval(target_model)
            frozen_trainable = sum(int(param.requires_grad) for param in target_model.encoder.parameters())
            print(f'CNN 骨干网络已冻结，encoder trainable params={frozen_trainable}')

    # Conv1 Finetuning: 当启用 conv1_ft 或从已训练模型继续训练时，解冻 conv1 并进入自蒸馏微调模式。
    # train_conv1: 从零开始训练 conv1，不需要自蒸馏
    # 检查是否需要解冻 conv1
    need_unfreeze_conv1 = bool(args.conv1_ft or args.train_conv1)
    
    # 启用 conv1 训练
    if need_unfreeze_conv1:
        model.encoder.enable_conv1_finetuning()

    # 定义模式变量
    conv1_ft_mode = bool((args.load_path and not args.load_weights_only) or args.conv1_ft)
    train_conv1_mode = bool(args.train_conv1)

    teacher_model = None
    if args.enable_feature_constancy:
        # 以 feature_constancy 为主
        teacher_ckpt = args.feature_constancy_ckpt.strip() if isinstance(args.feature_constancy_ckpt, str) else ''
        if teacher_ckpt == '':
            teacher_ckpt = args.load_path.strip() if isinstance(args.load_path, str) else ''
        if teacher_ckpt == '':
            raise RuntimeError('enable_feature_constancy=True 但 feature_constancy_ckpt/load_path 均为空，无法构建教师模型。')
        if not os.path.isfile(teacher_ckpt):
            raise RuntimeError(f'feature_constancy teacher checkpoint not found: {teacher_ckpt}')

        teacher_model = FingerprintNet(
            feat_dim=128,
            num_layers=ctrl.num_layers,
            knn_k=ctrl.knn_k,
            topk_pool=ctrl.topk_pool,
            attn_dropout=ctrl.attn_dropout,
            fps_alpha=ctrl.fps_alpha,
            tensor_neurons=ctrl.tensor_neurons,
            dropout_rate=ctrl.dropout_rate,
            gnn_dropout=ctrl.gnn_dropout,
            drop_edge_prob=ctrl.drop_edge_prob,
            feature_jitter=ctrl.feature_jitter,
            ntn_temperature=ctrl.ntn_temperature,
            l2_temperature=ctrl.l2_temperature,
            use_ntn=ctrl.use_ntn,
            disable_topology_score=ctrl.disable_topology_score,
            graph_backbone=ctrl.graph_backbone,
        ).to(device)
        _ = load_checkpoint_into_model(teacher_model, teacher_ckpt, device=device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(
            '[FeatureConstancy] enabled=True '
            f'| weight={float(args.feature_constancy_weight):.6f} '
            f'| teacher_ckpt={teacher_ckpt}'
        )
    else:
        # 只有在 feature_constancy 未启用时才处理 conv1_ft
        if conv1_ft_mode:
            teacher_ckpt = args.load_path.strip() if isinstance(args.load_path, str) else ''
            if teacher_ckpt == '':
                raise RuntimeError('conv1 微调模式需要提供 load_path 作为自蒸馏教师模型。')
            if not os.path.isfile(teacher_ckpt):
                raise RuntimeError(f'self-distillation teacher checkpoint not found: {teacher_ckpt}')

            teacher_model = FingerprintNet(
                feat_dim=128,
                num_layers=ctrl.num_layers,
                knn_k=ctrl.knn_k,
                topk_pool=ctrl.topk_pool,
                attn_dropout=ctrl.attn_dropout,
                fps_alpha=ctrl.fps_alpha,
                tensor_neurons=ctrl.tensor_neurons,
                dropout_rate=ctrl.dropout_rate,
                gnn_dropout=ctrl.gnn_dropout,
                drop_edge_prob=ctrl.drop_edge_prob,
                feature_jitter=ctrl.feature_jitter,
                ntn_temperature=ctrl.ntn_temperature,
                l2_temperature=ctrl.l2_temperature,
                use_ntn=ctrl.use_ntn,
                disable_topology_score=ctrl.disable_topology_score,
                graph_backbone=ctrl.graph_backbone,
            ).to(device)
            _ = load_checkpoint_into_model(teacher_model, teacher_ckpt, device=device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            print(
                '[SelfDistill] enabled=True '
                f'| teacher_ckpt={teacher_ckpt} '
                '| kd_weight: linear decay 1.0 -> 0.2 (first 80% epochs)'
            )
        elif train_conv1_mode:
            print('[TrainConv1] enabled=True | 从零开始训练 conv1 层，不使用自蒸馏')
        else:
            print('[FeatureConstancy] enabled=False')

    _apply_trainable_policy(model)
    print('[TrainPolicy] Parameter statistics after applying trainable policy:')
    count_model_parameters(model, verbose=True)

    pair_weight_logit = None
    pair_weight_optimizer_lr = 0.0
    criterion_arc = None
    if ctrl.enable_pair_loss and ctrl.enable_dynamic_pair_weight:
        dynamic_max = max(1e-6, float(ctrl.dynamic_pair_weight_max))
        dynamic_init = float(ctrl.dynamic_pair_weight_init)
        dynamic_init = min(max(dynamic_init, 1e-6), dynamic_max - 1e-6)
        dynamic_ratio = dynamic_init / dynamic_max
        dynamic_init_logit = math.log(dynamic_ratio / (1.0 - dynamic_ratio))
        pair_weight_logit = torch.nn.Parameter(
            torch.tensor(dynamic_init_logit, dtype=torch.float32, device=device)
        )
        if loaded_ckpt is not None and 'dynamic_pair_weight_logit' in loaded_ckpt:
            try:
                pair_weight_logit.data.fill_(float(loaded_ckpt['dynamic_pair_weight_logit']))
                print('[Resumer] Restored dynamic_pair_weight_logit from checkpoint.')
            except Exception as e:
                print(f'[Resumer] [Warn] dynamic_pair_weight_logit restore failed, keep init value: {e}')

        pair_weight_optimizer_lr = max(0.0, float(ctrl.lr) * float(ctrl.dynamic_pair_weight_lr_scale))
        dynamic_curr = float(torch.sigmoid(pair_weight_logit.detach()).item()) * dynamic_max
        print(
            '[DynamicPairWeight] enabled=True '
            f'| pre_hard={float(ctrl.dynamic_pair_weight_pre_hard):.4f} '
            f'| init={dynamic_curr:.4f} '
            f'| max={dynamic_max:.4f} '
            f'| lr={pair_weight_optimizer_lr:.8f}'
        )
    elif ctrl.enable_pair_loss:
        print('[DynamicPairWeight] enabled=False (compat mode, use static pair_loss_weight).')
    else:
        print('[PairLoss] disabled.')



    mixed_static_weight_unsw = float(args.mixed_static_weight_unsw)
    mixed_static_weight_cfpose = float(args.mixed_static_weight_cfpose)
    mixed_static_domain_weight_enabled = bool(
        args.data_name == 'mixed'
        and (
            abs(mixed_static_weight_unsw - 1.0) > 1e-8
            or abs(mixed_static_weight_cfpose - 1.0) > 1e-8
        )
    )

    mixed_domain_log_var_unsw = None
    mixed_domain_log_var_cfpose = None
    mixed_domain_weight_enabled = bool(args.data_name == 'mixed' and ctrl.enable_mixed_dynamic_domain_weight)
    mixed_domain_split_enabled = bool(
        args.data_name == 'mixed' and (mixed_domain_weight_enabled or mixed_static_domain_weight_enabled)
    )
    mixed_domain_lr = 0.0
    if mixed_domain_weight_enabled:
        init_log_var = float(ctrl.mixed_domain_weight_init_log_var)
        mixed_domain_log_var_unsw = torch.nn.Parameter(torch.tensor(init_log_var, device=device))
        mixed_domain_log_var_cfpose = torch.nn.Parameter(torch.tensor(init_log_var, device=device))
        mixed_domain_lr = max(float(ctrl.lr) * float(ctrl.mixed_domain_weight_lr_scale), 1e-8)
        print(
            '[MixedDomainWeight] enabled=True '
            f'| init_log_var={init_log_var:.4f} '
            f'| lr={mixed_domain_lr:.8f}'
        )
    else:
        print('[MixedDomainWeight] enabled=False (only affects mixed mode when explicitly enabled).')

    if args.data_name == 'mixed':
        print(
            '[MixedDomainWeight][Static] '
            f'unsw={mixed_static_weight_unsw:.3f}, cfpose={mixed_static_weight_cfpose:.3f}, '
            f'enabled={mixed_static_domain_weight_enabled}'
        )

    dynamic_pair_tail_threshold = float(ctrl.pair_tail_threshold)
    if ctrl.enable_pair_loss and ctrl.enable_dynamic_pair_tail_threshold and loaded_ckpt is not None:
        ckpt_tail = loaded_ckpt.get('dynamic_pair_tail_threshold', None)
        if ckpt_tail is not None:
            try:
                dynamic_pair_tail_threshold = float(ckpt_tail)
                print(f'[Resumer] Restored dynamic_pair_tail_threshold={dynamic_pair_tail_threshold:.4f} from checkpoint.')
            except Exception as e:
                print(f'[Resumer] [Warn] dynamic_pair_tail_threshold restore failed, use default: {e}')

    if ctrl.enable_pair_loss and ctrl.enable_dynamic_pair_tail_threshold:
        print(
            '[DynamicPairTailThreshold] enabled=True '
            f'| q={float(ctrl.dynamic_pair_tail_quantile):.3f} '
            f'| min={float(ctrl.dynamic_pair_tail_min):.3f} '
            f'| max={float(ctrl.dynamic_pair_tail_max):.3f} '
            f'| momentum={float(ctrl.dynamic_pair_tail_momentum):.3f} '
            f'| start={dynamic_pair_tail_threshold:.3f}'
        )
    elif ctrl.enable_pair_loss:
        print('[DynamicPairTailThreshold] enabled=False (compat mode, use static pair_tail_threshold).')

    def _attach_dynamic_pair_param_group(opt):
        if not ctrl.enable_pair_loss or pair_weight_logit is None:
            return
        if pair_weight_optimizer_lr <= 0.0:
            print('[DynamicPairWeight] [Warn] lr<=0, skip optimizer param group attach.')
            return
        opt.add_param_group(
            {
                'params': [pair_weight_logit],
                'lr': pair_weight_optimizer_lr,
                'weight_decay': 0.0,
            }
        )
        print('[DynamicPairWeight] optimizer param group attached.')




    optimizer = build_optimizer(model, ctrl)
    
    # Conv1 Finetuning: 重新构建优化器以实现判别式学习率
    if conv1_ft_mode:
        base_params = []
        conv1_params = []
        scorer_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder.backbone.0" in n or "encoder.layer0.0" in n:
                conv1_params.append(p)
            elif "scorer" in n or "fusion_w" in n:
                scorer_params.append(p)
            else:
                base_params.append(p)
        
        conv1_lr = args.lr * 0.1
        param_groups = [
            {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': scorer_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': conv1_params, 'lr': conv1_lr, 'weight_decay': args.weight_decay * 10.0},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        print(f'[Optimizer] Conv1 FT Enabled. Conv1 LR: {conv1_lr:.2e}, WD: {args.weight_decay * 10.0:.2e}')
    elif train_conv1_mode:
        # TrainConv1: 为conv1层设置较高的学习率
        base_params = []
        conv1_params = []
        scorer_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder.backbone.0" in n or "encoder.layer0.0" in n:
                conv1_params.append(p)
            elif "scorer" in n or "fusion_w" in n:
                scorer_params.append(p)
            else:
                base_params.append(p)
        
        conv1_lr = args.lr * 1.0  # 从零开始训练时，conv1使用与其他层相同的学习率
        param_groups = [
            {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': scorer_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': conv1_params, 'lr': conv1_lr, 'weight_decay': args.weight_decay},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        print(f'[Optimizer] TrainConv1 Enabled. Conv1 LR: {conv1_lr:.2e}, WD: {args.weight_decay:.2e}')
    
    _attach_dynamic_pair_param_group(optimizer)
    # 添加混合域权重参数组
    if mixed_domain_weight_enabled and mixed_domain_lr > 0.0:
        optimizer.add_param_group({'params': [mixed_domain_log_var_unsw, mixed_domain_log_var_cfpose], 'lr': mixed_domain_lr})
        print('[MixedDomainWeight] optimizer param group attached.')
    
    # 添加 ArcFace 参数组
    if criterion_arc is not None:
        optimizer.add_param_group({
            'params': criterion_arc.parameters(),
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        })
        print(f'[ArcFace] optimizer param group attached. LR: {args.lr:.2e}')
    
    optimizer_state_to_restore = None

    # Resume optimizer and epoch state if checkpoint was loaded
    # start_epoch 和 global_step 已经在前面初始化
    
    # 逻辑优化：从现有 Checkpoint 目录中递归搜寻“历史最佳轮次”
    # 这样即使崩溃重启，也能正确判断当前是否处于“性能高峰期”之后，从而决定是否跳过 Full EER
    def get_historical_best_epoch(ckpt_mgr):
        if ckpt_mgr.best:
            return ckpt_mgr.best[0].get('epoch', -1)
        return -1

    last_best_epoch = get_historical_best_epoch(ckpt_mgr)

    if loaded_ckpt and (not args.load_weights_only):
        if 'optimizer' in loaded_ckpt:
            optimizer_state_to_restore = loaded_ckpt['optimizer']
        if 'epoch' in loaded_ckpt and not args.conv1_ft:  # 只有非 conv1_ft 模式才从这里恢复 epoch
            start_epoch = loaded_ckpt['epoch']
            # If starting from a finished epoch, we start at next one in range
            print(f'[Resumer] Resuming training from epoch {start_epoch + 1}') 
        if 'global_step' in loaded_ckpt and not args.conv1_ft:  # 只有非 conv1_ft 模式才从这里恢复 global_step
            global_step = loaded_ckpt['global_step']
        rng_restored = False
        if 'rng_state' in loaded_ckpt:

            def _restore_rng_state(state):
                """Restore python/numpy/torch/cuda RNG states from checkpoint payload."""
                if not isinstance(state, dict):
                    return False, 'invalid rng_state payload'
                try:
                    if 'python' in state:
                        random.setstate(state['python'])
                    if 'numpy' in state:
                        np.random.set_state(state['numpy'])
                    if 'torch' in state:
                        torch.set_rng_state(state['torch'])
                    if torch.cuda.is_available() and ('cuda' in state) and (state['cuda'] is not None):
                        torch.cuda.set_rng_state_all(state['cuda'])
                    return True, ''
                except Exception as exc:
                    return False, str(exc)

            rng_restored, rng_err = _restore_rng_state(loaded_ckpt.get('rng_state'))
            if rng_restored:
                print('[Resumer] RNG states restored: python/numpy/torch/cuda.')
            else:
                print(f'[Resumer] [Warn] RNG state restore failed: {rng_err}')
        print('[Resumer] Restored states: model, optimizer(if compatible), epoch/global_step, phase_manager(if available).')
        if not rng_restored:
            print('[Resumer] Not restored: RNG states (python/numpy/torch/cuda), dataloader worker runtime state.')
        else:
            print('[Resumer] Not restored: dataloader worker runtime state.')
    elif loaded_ckpt and args.load_weights_only:
        print('[Resumer] Weights-only initialization completed. Start training from epoch 1 with fresh optimizer/phase state.')

    if optimizer_state_to_restore is not None:
        try:
            optimizer.load_state_dict(optimizer_state_to_restore)
            print(f'[Resumer] Optimizer state loaded from {args.load_path}')
        except Exception as e:
            print(
                '[Resumer] [Warn] Could not load optimizer state after stage-aware optimizer build '
                f'(likely due to phase/param-group mismatch): {e}'
            )

    model.train()
    _enforce_frozen_modules_eval(model)

    # 早停机制初始化
    best_val_eer = float('inf')
    early_stop_counter = 0
    early_stop_patience = args.early_stop_patience
    # 早停改进：添加容忍度、平滑和冷却期
    early_stop_tolerance = 1e-3  # 容忍度，当EER恶化小于此值时不增加计数器
    early_stop_min_improvement = 1e-3  # 最小改进阈值，只有超过此值才认为有改进
    val_eer_history = []  # 存储历史验证EER，用于移动平均
    moving_average_window = 3  # 移动平均窗口大小
    early_stop_cooling_period = 5  # 冷却期，前N次验证不触发早停
    validation_count = 0  # 独立的验证次数计数器，用于冷却期计算
    early_stop_triggered = False  # 早停触发标志

    def _worker_init_fn_pdeathsig(worker_id):
        """
        Prevent dataloader workers from becoming zombies if main process is killed (e.g. OOM Killer).
        Sets the child process to receive SIGKILL when its parent dies.
        """
        try:
            import ctypes
            import signal
            PR_SET_PDEATHSIG = 1
            ctypes.CDLL('libc.so.6').prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except Exception:
            pass
    
    libc_handle = None
    if bool(args.enable_malloc_trim):
        try:
            libc_handle = ctypes.CDLL('libc.so.6')
        except Exception:
            libc_handle = None
    
    def _release_process_memory(do_cuda_cache_clear: bool = False):
        import gc
        gc.collect()
        if do_cuda_cache_clear and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if libc_handle is not None:
            try:
                libc_handle.malloc_trim(0)
            except Exception:
                pass
    
    def _get_rss_mb() -> float:
        try:
            with open('/proc/self/status', 'r', encoding='utf-8') as status_f:
                for line in status_f:
                    if line.startswith('VmRSS:'):
                        rss_kb = float(line.split()[1])
                        return rss_kb / 1024.0
        except Exception:
            pass
        return float('nan')
    
    if ctrl.augment:
        print(
            '[Augment] enabled=True '
            f'| profile={args.augment_profile} '
            f'| micro_rotate={float(args.micro_aug_max_rotate):.2f} '
            f'| micro_translate_ratio={float(args.micro_aug_max_translate_ratio):.4f}'
        )
        if args.enable_random_illum_gradient:
            print(
                '[Augment][IllumGradient] enabled=True '
                f'| prob={float(args.illum_gradient_prob):.3f} '
                f'| strength=[{float(args.illum_gradient_strength_min):.3f}, '
                f'{float(args.illum_gradient_strength_max):.3f}] '
                f'| gamma=[{float(args.illum_gradient_gamma_min):.3f}, '
                f'{float(args.illum_gradient_gamma_max):.3f}]'
            )
        else:
            print('[Augment][IllumGradient] enabled=False')
    else:
        print('[Augment] enabled=False')

    # Balanced PK sampling training
    train_ds = FingerprintDataset(
        None,
        patch_size=32,
        stride=16,
        mode='train',
        files=train_files,
        augment=ctrl.augment,
        data_name=args.data_name,
        augment_profile=args.augment_profile,
        micro_rotate_deg=float(args.micro_aug_max_rotate),
        micro_translate_ratio=float(args.micro_aug_max_translate_ratio),
        enable_random_illum_gradient=bool(args.enable_random_illum_gradient),
        illum_gradient_prob=float(args.illum_gradient_prob),
        illum_gradient_strength_min=float(args.illum_gradient_strength_min),
        illum_gradient_strength_max=float(args.illum_gradient_strength_max),
        illum_gradient_gamma_min=float(args.illum_gradient_gamma_min),
        illum_gradient_gamma_max=float(args.illum_gradient_gamma_max),
    )

    # Dataset for Validation
    val_ds = FingerprintDataset(None, patch_size=32, stride=16, mode='test', 
                                files=val_files, data_name=args.data_name)
    val_loader = data_cls.DataLoader(val_ds, batch_size=args.benchmark_batch_size, shuffle=False, 
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     prefetch_factor=4 if args.num_workers > 0 else None,
                                     worker_init_fn=_worker_init_fn_pdeathsig)

    # Dataset for Benchmark (Used for full EER calculation)
    benchmark_ds = FingerprintDataset(None, patch_size=32, stride=16, mode='test', 
                                files=test_files, data_name=args.data_name)
    benchmark_loader = data_cls.DataLoader(benchmark_ds, batch_size=args.benchmark_batch_size, shuffle=False, 
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     prefetch_factor=4 if args.num_workers > 0 else None,
                                     worker_init_fn=_worker_init_fn_pdeathsig)
    
    # Get labels for sampler
    train_labels = []
    train_domains = []
    for fp in train_ds.files:
        file_data_name = train_ds._get_data_name_for_file(fp)
        train_labels.append(get_finger_id(fp, file_data_name))
        train_domains.append(file_data_name)

    if args.data_name == 'mixed':
        unsw_ids = set()
        cfpose_ids = set()
        invalid_id_count = 0
        for sample_id, domain in zip(train_labels, train_domains):
            sid = int(sample_id)
            if sid < 0:
                invalid_id_count += 1
                continue
            if domain == 'unsw':
                unsw_ids.add(sid)
            elif domain == 'cfpose':
                cfpose_ids.add(sid)

        overlap = unsw_ids.intersection(cfpose_ids)
        if overlap:
            overlap_preview = sorted(list(overlap))[:10]
            raise RuntimeError(
                '[Data] Mixed ID collision detected between UNSW and CFPose. '
                f'overlap_count={len(overlap)}, sample={overlap_preview}'
            )

        print(
            '[Data] Mixed ID collision check passed '
            f'| unsw_ids={len(unsw_ids)} | cfpose_ids={len(cfpose_ids)} | overlap=0'
        )
        if invalid_id_count > 0:
            print(f'[Data][Warn] mixed train set contains invalid ids: {invalid_id_count}')

    # ArcFace ID 映射：将原始 ID 转换为连续的 0~num_classes-1
    unique_ids = sorted(list(set(train_labels)))
    num_classes = len(unique_ids)
    id_to_idx = {orig_id: i for i, orig_id in enumerate(unique_ids)}
    print(f'[ArcFace] ID mapping created: {num_classes} unique IDs')

    # 初始化 ArcFace 层
    if getattr(args, 'enable_arcface', False):
        from MyModel.model.mymodel import ArcMarginProduct
        criterion_arc = ArcMarginProduct(
            in_features=128,
            out_features=num_classes,
            s=float(args.arcface_s),
            m=float(args.arcface_m),
        ).to(device)
        print(f'[ArcFace] ArcMarginProduct initialized: s={args.arcface_s}, m={args.arcface_m}')

    k_samples = 4

    def _build_train_loader(effective_batch_size: int, epoch_seed: int):
        p_ids = max(int(effective_batch_size) // k_samples, 1)
        n_steps = max(len(train_ds) // int(effective_batch_size), 1)
        sampler_domain_labels = train_domains if (args.data_name == 'mixed' or conv1_ft_mode or train_conv1_mode) else None
        sampler_unsw_ratio = 0.5 if ((conv1_ft_mode or train_conv1_mode) and args.data_name == 'mixed') else float(args.mixed_unsw_sampling_ratio)
        if (conv1_ft_mode or train_conv1_mode) and args.data_name == 'mixed':
            print('[Sampler] Conv1 mode: force domain_labels=train_domains and unsw_sampling_ratio=0.5')
        sampler = BalancedBatchSampler(
            labels=train_labels,
            n_classes=p_ids,
            n_samples=k_samples,
            num_batches=n_steps,
            seed=epoch_seed,
            domain_labels=sampler_domain_labels,
            unsw_sampling_ratio=sampler_unsw_ratio,
        )
        loader = data_cls.DataLoader(
            train_ds,
            batch_sampler=sampler,  # 使用 batch_sampler 替代 batch_size 和 shuffle
            num_workers=args.num_workers,
            collate_fn=collate_list,
            pin_memory=args.pin_memory,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=True if args.num_workers > 0 else False,
            worker_init_fn=_worker_init_fn_pdeathsig
        )
        return sampler, loader, p_ids, n_steps

    # 初始化阶段管理器（固定阶段长度）- 移到参数加载之后
    phase_manager = SimplePhaseManager(
        args.warm_up_epochs, args.batch_all_epochs, args.mixed_hard_epochs
    )

    if loaded_ckpt and (not args.load_weights_only) and 'phase_manager_state' in loaded_ckpt:
        try:
            phase_manager.load_state_dict(loaded_ckpt['phase_manager_state'])
            print(f'[Resumer] PhaseManager state loaded. Current Phase: {phase_manager.current_phase}, Epoch: {phase_manager.phase_epoch}')
        except Exception as e:
            print(f'[Resumer] [Warn] Could not load phase manager state: {e}')

    if conv1_ft_mode:
        phase_manager.current_phase = 3
        phase_manager.phase_epoch = 0
        args.hard_neg_quantile = 0.1
        print('[Evolution] Fine-tuning mode detected. Force skipping all phases. Starting at Phase 3 with q=0.1')
    elif train_conv1_mode:
        # TrainConv1 mode: start from Phase 0
        print('[Evolution] TrainConv1 mode detected. Starting from Phase 0.')

    # ArcFace 先锁后放策略：前 5 个 epoch 冻结 encoder，只训练 ArcFace
    arcface_warmup_epochs = 5
    if getattr(args, 'enable_arcface', False) and conv1_ft_mode:
        print(f'[ArcFace] Warm-up strategy enabled: freeze encoder for first {arcface_warmup_epochs} epochs')
        for param in model.encoder.parameters():
            param.requires_grad = False
        print('[ArcFace] Encoder frozen, only ArcFace layer will be trained initially')

    # 构建训练数据加载器
    train_sampler, train_loader, p_identities, steps_per_epoch = _build_train_loader(
        args.batch_size,
        (args.seed if hasattr(args, 'seed') else 0) + start_epoch,
    )

    last_eval_epoch = -1
    last_eval_cache = {
        'eval_loss': float('nan'),
        'eval_acc': float('nan'),
        'eval_eer': float('nan'),
        'eval_threshold': float('nan'),
    }
    epoch_eval_eer = float('nan')

    for epoch in range(start_epoch, args.epochs):
        if args.log_ram_usage:
            writer.add_scalar('System/RAM_RSS_MB/EpochStart', _get_rss_mb(), epoch)
        
        # Keep epoch-wise sampling deterministic but non-repeating across epochs.
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        epoch_train_loss = float('nan')
        epoch_train_acc = float('nan')
        eval_metric_updated_this_epoch = False

        kd_weight = 0.0
        if conv1_ft_mode and teacher_model is not None:
            decay_progress = min(1.0, epoch / max(1, int(args.epochs * 0.8)))
            kd_weight = 1.0 - (1.0 - 0.2) * decay_progress
            writer.add_scalar('Hyperparameters/kd_weight', kd_weight, epoch)

        # ArcFace 先锁后放策略：epoch 5 时解冻 encoder
        if getattr(args, 'enable_arcface', False) and conv1_ft_mode and epoch == arcface_warmup_epochs:
            print(f'[ArcFace] Epoch {epoch+1}: Thawing encoder for full fine-tuning')
            for param in model.encoder.parameters():
                param.requires_grad = True
            model.encoder.enable_conv1_finetuning()
            print('[ArcFace] Encoder thawed, full network fine-tuning begins')

        # --- Training Strategy (Dynamic Phase Transition) ---
        curr_phase = phase_manager.current_phase
        phase_epoch = phase_manager.phase_epoch
        T_curr_max = max(1, float(phase_manager.max_epochs[curr_phase] if phase_manager.max_epochs[curr_phase] != float('inf') else 1))
        
        # 计算全局进度以平滑过渡 margin（默认保持历史行为）
        progress = min(1.0, (curr_phase + min(1.0, phase_epoch / T_curr_max)) / 3.0)
        margin_factor = 0.25 + 0.75 * (0.5 * (1 - math.cos(math.pi * progress)))

        # 纯 Batch-Hard 初期使用固定 margin

        curr_margin = ctrl.margin * margin_factor
        
        def _phase_strategy_values(phase_idx, local_phase_epoch):
            local_T_max = max(1, float(phase_manager.max_epochs[phase_idx] if phase_manager.max_epochs[phase_idx] != float('inf') else 1))
            if phase_idx == 0:
                # 从0.3开始，而不是0，让模型更早开始学习有效的特征表示
                alpha = 0.3 + 0.7 * 0.5 * (1 - math.cos(math.pi * local_phase_epoch / local_T_max))
                return {
                    'w_triplet': alpha,
                    'w_all': 1.0,
                    'w_hard': 0.0,
                    'name': 'Warm-up'
                }
            if phase_idx == 1:
                return {
                    'w_triplet': 1.0,
                    'w_all': 1.0,
                    'w_hard': 0.0,
                    'name': 'Batch-All'
                }
            if phase_idx == 2:
                beta = min(1.0, local_phase_epoch / local_T_max)
                return {
                    'w_triplet': 1.0,
                    'w_all': 1.0 - beta,
                    'w_hard': beta,
                    'name': 'Mixed-Hard'
                }
            return {
                'w_triplet': 1.0,
                'w_all': 0.0,
                'w_hard': 1.0,
                'name': 'Batch-Hard'
            }

        base_strategy = _phase_strategy_values(curr_phase, phase_epoch)
        curr_w_triplet = float(base_strategy['w_triplet'])
        w_all = float(base_strategy['w_all'])
        w_hard = float(base_strategy['w_hard'])
        phase_name = f"{base_strategy['name']}, PhaseEpoch:{phase_epoch}, Margin:{curr_margin:.3f}"

        smoothing_factor = 1.0

        if ctrl.enable_pair_loss:
            if ctrl.enable_dynamic_pair_weight:
                curr_pair_mode = 'fixed_pre_hard'
                curr_pair_weight_epoch = float(ctrl.dynamic_pair_weight_pre_hard)
                if curr_phase == 3 and pair_weight_logit is not None:
                    curr_pair_mode = 'learnable_batch_hard'
                    curr_pair_weight_epoch = float(ctrl.dynamic_pair_weight_max) * float(
                        torch.sigmoid(pair_weight_logit.detach()).item()
                    )
            else:
                curr_pair_mode = 'static'
                curr_pair_weight_epoch = float(ctrl.pair_loss_weight)

            if ctrl.enable_dynamic_pair_tail_threshold:
                curr_pair_tail_threshold_epoch = float(dynamic_pair_tail_threshold)
                curr_pair_tail_mode = 'dynamic_quantile'
            else:
                curr_pair_tail_threshold_epoch = float(ctrl.pair_tail_threshold)
                curr_pair_tail_mode = 'static'

            print(
                f"--- Epoch {epoch+1}: {phase_name}, w_triplet={curr_w_triplet:.3f}, "
                f"w_all={w_all:.3f}, w_hard={w_hard:.3f}, pair_w={curr_pair_weight_epoch:.3f}({curr_pair_mode}), "
                f"pair_tail_th={curr_pair_tail_threshold_epoch:.3f}({curr_pair_tail_mode}) ---"
            )
        else:
            curr_pair_weight_epoch = 0.0
            curr_pair_mode = 'disabled'
            curr_pair_tail_threshold_epoch = 0.0
            curr_pair_tail_mode = 'disabled'
            print(
                f"--- Epoch {epoch+1}: {phase_name}, w_triplet={curr_w_triplet:.3f}, "
                f"w_all={w_all:.3f}, w_hard={w_hard:.3f}, pair_loss=disabled ---"
            )
        
        # 记录学习率到 TensorBoard（区分不同参数组）
        writer.add_scalar('Hyperparameters/w_triplet', curr_w_triplet, epoch)
        writer.add_scalar('Hyperparameters/phase_index', float(curr_phase), epoch)
        writer.add_scalar('Hyperparameters/margin_progress', float(progress), epoch)
        writer.add_scalar('Hyperparameters/margin', curr_margin, epoch)
        writer.add_scalar('Hyperparameters/margin_factor', margin_factor, epoch)
        writer.add_scalar('Hyperparameters/hard_neg_quantile', float(args.hard_neg_quantile), epoch)
        writer.add_scalar('Hyperparameters/ntn_reg_lambda', float(args.ntn_reg_lambda), epoch)
        if ctrl.enable_pair_loss:
            writer.add_scalar('Hyperparameters/pair_loss_weight', float(ctrl.pair_loss_weight), epoch)
            writer.add_scalar('Hyperparameters/pair_loss_weight_effective', curr_pair_weight_epoch, epoch)
            writer.add_scalar('Hyperparameters/pair_loss_weight_mode_learnable', 1.0 if curr_pair_mode == 'learnable_batch_hard' else 0.0, epoch)
            if pair_weight_logit is not None:
                writer.add_scalar('Hyperparameters/dynamic_pair_weight_logit', float(pair_weight_logit.detach().item()), epoch)
            writer.add_scalar('Hyperparameters/pair_tail_threshold', float(ctrl.pair_tail_threshold), epoch)
            writer.add_scalar('Hyperparameters/pair_tail_threshold_effective', curr_pair_tail_threshold_epoch, epoch)
            writer.add_scalar('Hyperparameters/pair_tail_threshold_mode_dynamic', 1.0 if curr_pair_tail_mode == 'dynamic_quantile' else 0.0, epoch)
            writer.add_scalar('Hyperparameters/pair_tail_alpha', float(ctrl.pair_tail_alpha), epoch)
        writer.add_scalar('Ablation/use_ntn', 1.0 if ctrl.use_ntn else 0.0, epoch)
        writer.add_scalar('Ablation/disable_topology_score', 1.0 if ctrl.disable_topology_score else 0.0, epoch)
        writer.add_scalar('Ablation/use_gcn_backbone', 1.0 if ctrl.graph_backbone == 'gcn' else 0.0, epoch)
        writer.add_scalar('Ablation/mixed_dynamic_domain_weight', 1.0 if mixed_domain_weight_enabled else 0.0, epoch)
        writer.add_scalar('Ablation/mixed_static_domain_weight', 1.0 if mixed_static_domain_weight_enabled else 0.0, epoch)
        writer.add_scalar('Hyperparameters/mixed_static_weight_unsw', float(mixed_static_weight_unsw), epoch)
        writer.add_scalar('Hyperparameters/mixed_static_weight_cfpose', float(mixed_static_weight_cfpose), epoch)
        if ctrl.use_ntn and hasattr(model, 'scorer'):
            writer.add_scalar('Hyperparameters/fusion_alpha', torch.sigmoid(model.scorer.alpha).item(), epoch)

        import collections
        epoch_step_metrics = collections.defaultdict(list)
        attn_head_trainable = any(param.requires_grad for param in model.attn_head.parameters())
        if not attn_head_trainable:
            print(f"[Info][Epoch {epoch+1}] attention_grad_norm monitor skipped (attn_head is frozen in current stage).")

        from tqdm import tqdm
        dynamic_tail_impostor_scores = []
        dynamic_tail_impostor_seen = 0
        dynamic_tail_max_samples = max(1, int(ctrl.dynamic_pair_tail_max_samples))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            
            # batch 现在是 dict: {key: [B, ...]}
            optimizer.zero_grad(set_to_none=True)
            
            # 移动数据到设备
            batch = move_sample_to_device(batch, device)
            
            # --- 前向传播 ---
            if args.baseline_b:
                embeddings, logits_all = model.forward_baseline_b(batch)  # [B, feat_dim], [B, N]
            else:
                embeddings, logits_all = model(batch)  # [B, feat_dim], [B, N], dict

            loss_kd = embeddings.mean() * 0.0
            if teacher_model is not None:
                with torch.no_grad():
                    if args.baseline_b:
                        teacher_embeddings, _ = teacher_model.forward_baseline_b(batch)
                    else:
                        teacher_embeddings, _ = teacher_model(batch)
                student_norm = F.normalize(embeddings, p=2, dim=1)
                teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)
                loss_kd = F.mse_loss(student_norm, teacher_norm)

            # 记录第一个样本的 logits（用于日志）
            first_imp = logits_all[0]
            
            labels = batch['id'].to(device)  # [B] 确保在正确的设备上
            
            # 记录 embedding 的统计信息
            with torch.no_grad():
                epoch_step_metrics['Features/embedding_norm_mean'].append(embeddings.norm(dim=1).mean().item())
                epoch_step_metrics['Features/embedding_norm_std'].append(embeddings.norm(dim=1).std().item())
            
            # --- 使用 GRASP 的 NTN 打分器计算 Triplet Loss ---
            # 计算所有样本对之间的相似度分数 [B, B]
            # 使用 batch 化的方式计算相似度矩阵
            B = embeddings.size(0)
            
            # 应用特征随机掩码（仅在训练阶段）
            embeddings_for_sim = embeddings
            
            # 扩展 embeddings 以便 batch 计算所有样本对
            # feat1: [B, B, feat_dim] - 每行是相同的 embedding
            # feat2: [B, B, feat_dim] - 每列是相同的 embedding
            feat1 = embeddings_for_sim.unsqueeze(0).expand(B, B, -1)  # [B, B, D]
            feat2 = embeddings_for_sim.unsqueeze(1).expand(B, B, -1)  # [B, B, D]
            
            # Batch 计算所有样本对的相似度
            sim_matrix = compute_pairwise_similarity(
                model,
                feat1.reshape(B * B, -1),
                feat2.reshape(B * B, -1),
                use_ntn=ctrl.use_ntn,
            )
            sim_matrix = sim_matrix.view(B, B)  # [B, B]

            if ctrl.enable_pair_loss and ctrl.enable_dynamic_pair_tail_threshold:
                with torch.no_grad():
                    neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0)
                    eye = torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
                    neg_mask = neg_mask & (~eye)
                    if neg_mask.any() and dynamic_tail_impostor_seen < dynamic_tail_max_samples:
                        imp_scores_batch = sim_matrix.detach()[neg_mask].flatten().cpu().float()
                        remain = dynamic_tail_max_samples - dynamic_tail_impostor_seen
                        if imp_scores_batch.numel() > remain:
                            imp_scores_batch = imp_scores_batch[:remain]
                        if imp_scores_batch.numel() > 0:
                            dynamic_tail_impostor_scores.append(imp_scores_batch)
                            dynamic_tail_impostor_seen += int(imp_scores_batch.numel())
            
            # 收集 NTN 分数统计信息
            ntn_scores_all = [sim_matrix.detach()]
            
            # 记录 NTN 分数分布
            if len(ntn_scores_all) > 0:
                ntn_scores_flat = torch.cat(ntn_scores_all, dim=0)
                score_prefix = 'NTN' if ctrl.use_ntn else 'Fallback'
                with torch.no_grad():
                    epoch_step_metrics[f'{score_prefix}/score_mean'].append(ntn_scores_flat.mean().item())
                    epoch_step_metrics[f'{score_prefix}/score_std'].append(ntn_scores_flat.std().item())
                    epoch_step_metrics[f'{score_prefix}/score_min'].append(ntn_scores_flat.min().item())
                    epoch_step_metrics[f'{score_prefix}/score_max'].append(ntn_scores_flat.max().item())
                    # 直方图
                    if args.tb_verbose and batch_idx == len(train_loader) - 1:
                        writer.add_histogram(f'{score_prefix}/score_distribution', ntn_scores_flat.cpu().numpy(), epoch)
            
            # 将相似度转换为距离：distance = 1 - similarity
            dist_mat = 1.0 - sim_matrix
            
            # --- Triplet Loss Calculation (Progressive Logic) ---
            # 计算最难正样本距离用于监控
            B = dist_mat.size(0)
            labels_unsqueezed = labels.unsqueeze(0)  # [1, B]
            mask_pos = (labels_unsqueezed == labels_unsqueezed.transpose(0, 1)).float()  # [B, B]
            mask_pos = mask_pos * (1.0 - torch.eye(B, device=dist_mat.device))
            hard_pos, _ = torch.max(dist_mat * mask_pos, dim=1)
            hpm_max_pos = hard_pos.max().item()
            epoch_step_metrics['Triplet/HPM_Max_Pos'].append(hpm_max_pos)
            
            loss_triplet_global = compute_triplet_loss_by_phase(
                dist_mat,
                labels,
                curr_phase=curr_phase,
                curr_margin=curr_margin,
                w_all=w_all,
                w_hard=w_hard,
                hard_neg_quantile=float(args.hard_neg_quantile),
            )

            loss_triplet = loss_triplet_global
            loss_triplet_unsw = dist_mat.mean() * 0.0
            loss_triplet_cfpose = dist_mat.mean() * 0.0
            if mixed_domain_split_enabled:
                unsw_mask = labels >= 1000000
                cfpose_mask = ~unsw_mask
                unsw_valid = int(unsw_mask.sum().item()) > 1
                cfpose_valid = int(cfpose_mask.sum().item()) > 1

                if unsw_valid:
                    dist_unsw = dist_mat[unsw_mask][:, unsw_mask]
                    labels_unsw = labels[unsw_mask]
                    loss_triplet_unsw = compute_triplet_loss_by_phase(
                        dist_unsw,
                        labels_unsw,
                        curr_phase=curr_phase,
                        curr_margin=curr_margin,
                        w_all=w_all,
                        w_hard=w_hard,
                        hard_neg_quantile=float(args.hard_neg_quantile),
                    )

                if cfpose_valid:
                    dist_cfpose = dist_mat[cfpose_mask][:, cfpose_mask]
                    labels_cfpose = labels[cfpose_mask]
                    loss_triplet_cfpose = compute_triplet_loss_by_phase(
                        dist_cfpose,
                        labels_cfpose,
                        curr_phase=curr_phase,
                        curr_margin=curr_margin,
                        w_all=w_all,
                        w_hard=w_hard,
                        hard_neg_quantile=float(args.hard_neg_quantile),
                    )

                loss_terms = []
                if unsw_valid:
                    unsw_term = loss_triplet_unsw
                    if mixed_domain_weight_enabled:
                        unsw_term = torch.exp(-mixed_domain_log_var_unsw) * unsw_term + mixed_domain_log_var_unsw
                    unsw_term = mixed_static_weight_unsw * unsw_term
                    loss_terms.append(unsw_term)
                if cfpose_valid:
                    cfpose_term = loss_triplet_cfpose
                    if mixed_domain_weight_enabled:
                        cfpose_term = torch.exp(-mixed_domain_log_var_cfpose) * cfpose_term + mixed_domain_log_var_cfpose
                    cfpose_term = mixed_static_weight_cfpose * cfpose_term
                    loss_terms.append(cfpose_term)

                if len(loss_terms) > 0:
                    loss_triplet = sum(loss_terms)
                else:
                    loss_triplet = loss_triplet_global

            curr_pair_weight = torch.tensor(0.0, device=embeddings.device)
            pair_loss = torch.zeros((), device=embeddings.device)
            if ctrl.enable_pair_loss:
                if ctrl.enable_dynamic_pair_weight:
                    if curr_phase == 3 and pair_weight_logit is not None:
                        curr_pair_weight = float(ctrl.dynamic_pair_weight_max) * torch.sigmoid(pair_weight_logit)
                    else:
                        curr_pair_weight = torch.tensor(float(ctrl.dynamic_pair_weight_pre_hard), device=embeddings.device)
                else:
                    curr_pair_weight = torch.tensor(float(ctrl.pair_loss_weight), device=embeddings.device)

                if float(curr_pair_weight.detach().item()) > 0:
                    pair_loss = pair_tail_alignment_loss(
                        sim_matrix,
                        labels,
                        tail_threshold=curr_pair_tail_threshold_epoch,
                        tail_alpha=float(ctrl.pair_tail_alpha),
                        tail_tau=float(ctrl.pair_tail_tau),
                    )
            
            # ArcFace 分类损失
            loss_cls = torch.zeros((), device=embeddings.device)
            if criterion_arc is not None:
                mapped_labels = torch.tensor([id_to_idx[int(x)] for x in labels], device=device)
                output_arc = criterion_arc(embeddings, mapped_labels)
                loss_cls = F.cross_entropy(output_arc, mapped_labels)
                epoch_step_metrics['Train/loss_arcface'].append(loss_cls.item())

            # 综合 Loss
            if conv1_ft_mode and teacher_model is not None:
                loss = loss_triplet + float(ctrl.pair_loss_weight) * pair_loss + kd_weight * loss_kd
            else:
                loss = curr_w_triplet * loss_triplet + curr_pair_weight * pair_loss
                if teacher_model is not None:
                    loss = loss + float(args.feature_constancy_weight) * loss_kd
            
            # 添加 ArcFace 损失
            if criterion_arc is not None:
                arcface_weight = float(args.arcface_weight)
                loss = loss + arcface_weight * loss_cls
            
            if ctrl.use_ntn:
                # 添加融合权重的软正则化 (L2 Penalty)，缓解特征极其偏向单模态
                ntn_l2_reg = 0.0
                for name, param in model.named_parameters():
                    if "alpha" in name or "fusion_w" in name:
                        ntn_l2_reg += torch.norm(param) ** 2
                epoch_step_metrics['NTN/fusion_l2_reg'].append(ntn_l2_reg.item())
                reg_term = float(args.ntn_reg_lambda) * ntn_l2_reg
                loss = loss + reg_term
            
            loss.backward()
            
            # --- 采集梯度与权重范数 (用于诊断) ---
            with torch.no_grad():
                # 注意力头梯度范数 (排查是否陷入死区)
                if attn_head_trainable:
                    attn_grad_norm = 0.0
                    for param in model.attn_head.parameters():
                        if param.grad is not None:
                            attn_grad_norm += param.grad.data.norm(2).item() ** 2
                    attn_grad_norm = attn_grad_norm ** 0.5
                    epoch_step_metrics['Gradients/attention_grad_norm'].append(attn_grad_norm)
                
                # 融合参数(包括 NTN 和图级) 的 L2 范数
                fusion_l2_norm = 0.0
                if ctrl.use_ntn:
                    for param in model.scorer.parameters():
                        fusion_l2_norm += param.data.norm(2).item() ** 2
                fusion_l2_norm += model.fusion_w.data.norm(2).item() ** 2
                fusion_l2_norm = fusion_l2_norm ** 0.5
                epoch_step_metrics['Weights/fusion_l2_norm'].append(fusion_l2_norm)
                
            optimizer.step()

            # 使用 GRASP 的 NTN 打分器计算的距离矩阵
            with torch.no_grad():
                # dist_mat 已经在上面计算过了 (1.0 - sim_matrix)
                hard_pos, hard_neg, valid_anchor_mask = compute_batch_hard_stats_for_ntn(
                    dist_mat,
                    labels,
                    hard_neg_quantile=float(args.hard_neg_quantile),
                )
                if valid_anchor_mask.sum() > 0:
                    acc = (hard_pos[valid_anchor_mask] < hard_neg[valid_anchor_mask]).float().mean().item()

                    # 记录 hard 挖掘的有效样本比例与难样本几何间隔，便于定位“高位平台”根因。
                    valid_ratio = valid_anchor_mask.float().mean().item()
                    epoch_step_metrics['Triplet/valid_anchor_ratio'].append(valid_ratio)
                    epoch_step_metrics['Triplet/hard_pos_mean'].append(hard_pos[valid_anchor_mask].mean().item())
                    epoch_step_metrics['Triplet/hard_neg_mean'].append(hard_neg[valid_anchor_mask].mean().item())
                    epoch_step_metrics['Triplet/hard_gap_mean'].append((hard_neg[valid_anchor_mask] - hard_pos[valid_anchor_mask]).mean().item())
                else:
                    acc = 0.0
                    epoch_step_metrics['Triplet/valid_anchor_ratio'].append(0.0)

            running_loss += float(loss.item())
            running_acc += acc
            seen += 1
            global_step += 1

            # 提取和整合指标进行记录，去除基于 10 steps 采样的偏差
            with torch.no_grad():
                # 计算 sigmoid
                imp_sig = torch.sigmoid(first_imp)  # [N]
                if 'node_mask' in batch and batch['node_mask'] is not None:
                    # 获取第一个样本的有效节点 mask
                    valid_mask = batch['node_mask'][0]  # [N]
                    if valid_mask.any():
                        valid_logits = first_imp[valid_mask]  # [N_valid]
                        valid_imp = imp_sig[valid_mask]  # [N_valid]
                    else:
                        valid_logits = first_imp
                        valid_imp = imp_sig
                    masked_ratio = 1.0 - valid_mask.float().mean().item()
                    epoch_step_metrics['Train/attention_masked_ratio'].append(masked_ratio)
                else:
                    valid_logits = first_imp
                    valid_imp = imp_sig
                
                imp_sig_valid = valid_imp

                # 记录详细的损失分量
                epoch_step_metrics['Train/loss_triplet'].append(loss_triplet.item())
                if mixed_domain_split_enabled:
                    epoch_step_metrics['Train/loss_triplet_unsw'].append(loss_triplet_unsw.item())
                    epoch_step_metrics['Train/loss_triplet_cfpose'].append(loss_triplet_cfpose.item())
                    epoch_step_metrics['MixedDomain/static_weight_unsw'].append(float(mixed_static_weight_unsw))
                    epoch_step_metrics['MixedDomain/static_weight_cfpose'].append(float(mixed_static_weight_cfpose))
                if mixed_domain_weight_enabled:
                    epoch_step_metrics['MixedDomain/log_var_unsw'].append(mixed_domain_log_var_unsw.detach().item())
                    epoch_step_metrics['MixedDomain/log_var_cfpose'].append(mixed_domain_log_var_cfpose.detach().item())
                    epoch_step_metrics['MixedDomain/weight_unsw'].append(torch.exp(-mixed_domain_log_var_unsw.detach()).item())
                    epoch_step_metrics['MixedDomain/weight_cfpose'].append(torch.exp(-mixed_domain_log_var_cfpose.detach()).item())
                epoch_step_metrics['Train/loss_pair_tail'].append(pair_loss.item())
                if teacher_model is not None:
                    epoch_step_metrics['Train/loss_kd'].append(loss_kd.item())
                epoch_step_metrics['Train/loss_total'].append(loss.item())
                epoch_step_metrics['Train/accuracy_batch'].append(acc)
                
                # 记录注意力 logits 的统计信息
                epoch_step_metrics['Train/attention_logits_min'].append(valid_logits.min().item())
                epoch_step_metrics['Train/attention_logits_max'].append(valid_logits.max().item())
                epoch_step_metrics['Train/attention_logits_mean'].append(valid_logits.mean().item())
                if valid_logits.numel() > 1:
                    epoch_step_metrics['Train/attention_logits_std'].append(valid_logits.std().item())
                epoch_step_metrics['Train/attention_prob_mean'].append(imp_sig_valid.mean().item())
                if imp_sig_valid.numel() > 1:
                    epoch_step_metrics['Train/attention_prob_std'].append(imp_sig_valid.std().item())
                
                # 记录 NTN 模块的融合权重
                if ctrl.use_ntn:
                    fusion_alpha = torch.sigmoid(model.scorer.alpha).item()
                    epoch_step_metrics['NTN/fusion_alpha'].append(fusion_alpha)

                if args.tb_verbose and batch_idx == len(train_loader) - 1:
                    if ctrl.use_ntn:
                        writer.add_histogram('NTN/alpha_weights', model.scorer.alpha.cpu().detach().numpy(), epoch)
                    writer.add_histogram('Train/attention_logits', valid_logits.cpu().detach().numpy(), epoch)
                    writer.add_histogram('Train/attention_probs', imp_sig_valid.cpu().detach().numpy(), epoch)

            if (batch_idx + 1) % 50 == 0:
                strategy_name = phase_name
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/seen:.4f}',
                    'acc': f'{running_acc/seen:.3f}',
                    'l_trip': f'{loss_triplet.item():.4f}',
                    'l_tail': f'{pair_loss.item():.4f}',
                    'l_kd': f'{loss_kd.item():.4f}',
                    'margin': f'{curr_margin:.3f}',
                })
                
                pbar.write(f'Epoch {epoch+1}/{args.epochs} | Step {batch_idx+1}/{len(train_loader)} | loss={running_loss/seen:.4f} | acc={running_acc/seen:.3f} | phase={strategy_name}')
                pbar.write(f'  [Losses] l_triplet={loss_triplet.item():.4f} | l_tail={pair_loss.item():.4f} | l_kd={loss_kd.item():.4f} | margin={curr_margin:.3f}')
                pbar.write(f'  [Logits] min={valid_logits.min().item():.3f}, max={valid_logits.max().item():.3f}, mean={valid_logits.mean().item():.3f} | Prob_mean={imp_sig_valid.mean().item():.3f}')

        # 落盘聚合的指标信息
        for metric_name, values in epoch_step_metrics.items():
            if len(values) > 0:
                writer.add_scalar(metric_name, sum(values) / len(values), epoch)

        # 训练健康度告警：用于尽早发现“梯度归零 + 打分饱和”组合问题。
        avg_attn_grad = None
        avg_hard_gap = None
        if len(epoch_step_metrics['Gradients/attention_grad_norm']) > 0:
            avg_attn_grad = sum(epoch_step_metrics['Gradients/attention_grad_norm']) / len(epoch_step_metrics['Gradients/attention_grad_norm'])
            if avg_attn_grad < 1e-5:
                print(f"[Warn][Epoch {epoch+1}] attention_grad_norm 极低 ({avg_attn_grad:.2e})，注意力分支可能进入饱和区。")

        if len(epoch_step_metrics['Triplet/hard_gap_mean']) > 0:
            avg_hard_gap = sum(epoch_step_metrics['Triplet/hard_gap_mean']) / len(epoch_step_metrics['Triplet/hard_gap_mean'])

        score_prefix = 'NTN' if ctrl.use_ntn else 'Fallback'
        score_min_key = f'{score_prefix}/score_min'
        score_mean_key = f'{score_prefix}/score_mean'
        if len(epoch_step_metrics[score_min_key]) > 0 and len(epoch_step_metrics[score_mean_key]) > 0:
            avg_score_min = sum(epoch_step_metrics[score_min_key]) / len(epoch_step_metrics[score_min_key])
            avg_score_mean = sum(epoch_step_metrics[score_mean_key]) / len(epoch_step_metrics[score_mean_key])
            if avg_score_min > 0.98 and avg_score_mean > 0.99:
                print(
                    f"[Warn][Epoch {epoch+1}] {score_prefix} 分数疑似饱和 "
                    f"(mean={avg_score_mean:.4f}, min={avg_score_min:.4f})，"
                    "建议检查 hard 强度与 margin 节奏。"
                )

        pbar.close()

        if ctrl.enable_pair_loss and ctrl.enable_dynamic_pair_tail_threshold:
            if len(dynamic_tail_impostor_scores) > 0:
                all_imp_scores = torch.cat(dynamic_tail_impostor_scores, dim=0)
                q = min(max(float(ctrl.dynamic_pair_tail_quantile), 0.0), 1.0)
                raw_threshold = float(torch.quantile(all_imp_scores, q).item())
                tail_min = float(ctrl.dynamic_pair_tail_min)
                tail_max = float(ctrl.dynamic_pair_tail_max)
                if tail_min > tail_max:
                    tail_min, tail_max = tail_max, tail_min
                clipped_threshold = min(max(raw_threshold, tail_min), tail_max)
                mom = min(max(float(ctrl.dynamic_pair_tail_momentum), 0.0), 0.999)
                if mom > 0.0:
                    dynamic_pair_tail_threshold = mom * float(dynamic_pair_tail_threshold) + (1.0 - mom) * clipped_threshold
                else:
                    dynamic_pair_tail_threshold = clipped_threshold
                writer.add_scalar('Hyperparameters/pair_tail_threshold_next_epoch', dynamic_pair_tail_threshold, epoch)
                writer.add_scalar('Hyperparameters/pair_tail_threshold_quantile_raw', raw_threshold, epoch)
                writer.add_scalar('Hyperparameters/pair_tail_impostor_samples', float(dynamic_tail_impostor_seen), epoch)
                print(
                    f"[DynamicPairTailThreshold][Epoch {epoch+1}] q={q:.3f}, "
                    f"raw={raw_threshold:.4f}, clipped={clipped_threshold:.4f}, "
                    f"next={dynamic_pair_tail_threshold:.4f}, samples={dynamic_tail_impostor_seen}"
                )
            else:
                print(f"[DynamicPairTailThreshold][Epoch {epoch+1}] no impostor samples collected; keep threshold={dynamic_pair_tail_threshold:.4f}")

        print(f'== Epoch {epoch+1} done: loss={running_loss/seen:.4f} | triplet-acc={running_acc/seen:.3f}')
        epoch_train_loss = running_loss / max(seen, 1)
        epoch_train_acc = running_acc / max(seen, 1)
        
        if args.log_ram_usage:
            writer.add_scalar('System/RAM_RSS_MB/AfterTrainEpoch', _get_rss_mb(), epoch)

        # 记录训练指标到 TensorBoard
        writer.add_scalar('Train/loss', epoch_train_loss, epoch)
        writer.add_scalar('Train/accuracy', epoch_train_acc, epoch)
        
        # 记录融合权重的直方图
        if args.tb_verbose:
            with torch.no_grad():
                writer.add_histogram('Weights/fusion_w', model.fusion_w.cpu().numpy(), epoch)

        # 使用固定阶段长度进行切换
        total_epoch = epoch + 1
        if conv1_ft_mode:
            if total_epoch < args.batch_all_epochs:
                if phase_manager.current_phase != 1:
                    phase_manager._switch_phase(1, "Conv1 FT: force phase 1")
            elif total_epoch < args.batch_all_epochs + args.mixed_hard_epochs:
                if phase_manager.current_phase != 2:
                    phase_manager._switch_phase(2, "Conv1 FT: transition to mixed hard")
            else:
                if phase_manager.current_phase != 3:
                    phase_manager._switch_phase(3, "Conv1 FT: transition to batch hard")
        else:
            if total_epoch < args.warm_up_epochs:
                # Phase 0: warm up
                if phase_manager.current_phase != 0:
                    phase_manager._switch_phase(0, "Fixed phase switching: warm up phase")
            elif total_epoch < args.warm_up_epochs + args.batch_all_epochs:
                # Phase 1: batch all
                if phase_manager.current_phase != 1:
                    phase_manager._switch_phase(1, "Fixed phase switching: batch all phase")
            elif total_epoch < args.warm_up_epochs + args.batch_all_epochs + args.mixed_hard_epochs:
                # Phase 2: mixed hard
                if phase_manager.current_phase != 2:
                    phase_manager._switch_phase(2, "Fixed phase switching: mixed hard phase")
            else:
                # Phase 3: standard (batch hard)
                if phase_manager.current_phase != 3:
                    phase_manager._switch_phase(3, "Fixed phase switching: standard phase")


        # ---- checkpointing (end of epoch) ----
        def _capture_rng_state():
            """Capture python/numpy/torch/cuda RNG states for resumable training."""
            state = {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                state['cuda'] = torch.cuda.get_rng_state_all()
            return state
        payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': int(epoch + 1),
            'global_step': int(global_step),
            'phase_manager_state': phase_manager.state_dict(),
            'rng_state': _capture_rng_state(),
            'args': vars(args),
            'control': ctrl.to_dict(),
            'metrics': {
                'train_loss': float(epoch_train_loss),
                'train_acc': float(epoch_train_acc),
                'benchmark_eer': float(epoch_eval_eer),
            },
        }
        if ctrl.enable_pair_loss:
            payload['dynamic_pair_tail_threshold'] = float(dynamic_pair_tail_threshold)
            if pair_weight_logit is not None:
                payload['dynamic_pair_weight_logit'] = float(pair_weight_logit.detach().item())
        # 保存去相关权重参数

        ckpt_mgr.save_last(payload)

        # ---- 验证集评估 (Validation Evaluation) ----
        # 逻辑：每 5 个 epoch 执行一次验证集评估
        is_last_epoch = (epoch + 1 == args.epochs)
        if (epoch + 1) % args.val_every_n_epochs == 0 or is_last_epoch:
            eval_target_model = model
            eval_label = f"Test (Epoch {epoch+1})"

            print(f'== [Test] EER Evaluation on {eval_label} ...')
            
            if args.log_ram_usage:
                writer.add_scalar('System/RAM_RSS_MB/BeforePeriodicEval', _get_rss_mb(), epoch)
            
            # 在测试前显式释放训练产生的缓存和计算图，避免测试期 OOM
            _release_process_memory(do_cuda_cache_clear=True)
            
            # Ensure evaluation does not leak eval mode into the training model.
            was_training_mode = eval_target_model.training
            try:
                if args.data_name == 'mixed':
                    if args.only_cfpose_test:
                        print('== [Test] Mixed mode: only evaluating CFPose (skipping UNSW to save time)...')
                        
                        # 只执行 CFPose 测试
                        cfpose_eer, cfpose_thr, cfpose_tp, cfpose_tn, cfpose_fp, cfpose_fn, cfpose_tar_1e4, _rank1, _rank5, _rank10 = test_benchmark(
                            eval_target_model,
                            benchmark_loader,
                            device,
                            dataset_name='cfpose',
                            epoch_info=epoch+1,
                            ckpt_dir=args.ckpt_dir,
                            use_ntn=ctrl.use_ntn,
                            enable_score_export=False,
                            enable_det_curve=False,
                            enable_score_histogram=True,
                        )
                        
                        cfpose_pairs = cfpose_tp + cfpose_tn + cfpose_fp + cfpose_fn
                        
                        print(
                            f'== [Test] Mixed Result: '
                            f'CFPose EER={cfpose_eer*100:.2f}% (thr={cfpose_thr:.4f}, TAR@0.01%={cfpose_tar_1e4*100:.2f}%)'
                        )
                        
                        writer.add_scalar('Test/EER_CFPOSE', cfpose_eer, epoch)
                        writer.add_scalar('Test/EER_Threshold_CFPOSE', cfpose_thr, epoch)
                        writer.add_scalar('Test/TAR_at_FAR_0_01pct_CFPOSE', cfpose_tar_1e4, epoch)
                        writer.add_scalar('Test/Pairs_CFPOSE', float(cfpose_pairs), epoch)
                        
                        # 使用 CFPose 的 EER 作为 eval_eer
                        epoch_eval_eer = cfpose_eer
                        epoch_eval_threshold = float('nan')  # 混合模式没有单一阈值
                        eval_metric_updated_this_epoch = True
                        last_eval_epoch = epoch + 1
                        last_eval_cache = {
                            'eval_loss': float('nan'),
                            'eval_acc': float('nan'),
                            'eval_eer': float(epoch_eval_eer),
                            'eval_threshold': float(epoch_eval_threshold),
                        }
                        
                        # 保存混合数据集的 EER 结果，用于 checkpoint 命名
                        benchmark_eers = {
                            'unsw': float('nan'),
                            'cfpose': cfpose_eer
                        }
                        unsw_tar_1e4 = float('nan')
                    else:
                        print('== [Test] Mixed mode: evaluating UNSW and CFPose separately...')

                        unsw_eer, unsw_thr, unsw_tp, unsw_tn, unsw_fp, unsw_fn, unsw_tar_1e4, unsw_rank1, unsw_rank5, unsw_rank10 = test_benchmark(
                            eval_target_model,
                            benchmark_loader,
                            device,
                            dataset_name='unsw',
                            epoch_info=epoch+1,
                            ckpt_dir=args.ckpt_dir,
                            use_ntn=ctrl.use_ntn,
                            enable_score_export=False,
                            enable_det_curve=False,
                            enable_score_histogram=True,
                        )
                        cfpose_eer, cfpose_thr, cfpose_tp, cfpose_tn, cfpose_fp, cfpose_fn, cfpose_tar_1e4, _rank1, _rank5, _rank10 = test_benchmark(
                            eval_target_model,
                            benchmark_loader,
                            device,
                            dataset_name='cfpose',
                            epoch_info=epoch+1,
                            ckpt_dir=args.ckpt_dir,
                            use_ntn=ctrl.use_ntn,
                            enable_score_export=False,
                            enable_det_curve=False,
                            enable_score_histogram=True,
                        )

                        unsw_pairs = unsw_tp + unsw_tn + unsw_fp + unsw_fn
                        cfpose_pairs = cfpose_tp + cfpose_tn + cfpose_fp + cfpose_fn
                        total_pairs = unsw_pairs + cfpose_pairs

                        mixed_weighted_eer = float('nan')
                        if total_pairs > 0 and np.isfinite(unsw_eer) and np.isfinite(cfpose_eer):
                            mixed_weighted_eer = (unsw_eer * unsw_pairs + cfpose_eer * cfpose_pairs) / total_pairs
                        unsw_rank1_text = f'{unsw_rank1*100:.2f}%' if np.isfinite(unsw_rank1) else 'N/A'
                        unsw_rank5_text = f'{unsw_rank5*100:.2f}%' if np.isfinite(unsw_rank5) else 'N/A'
                        unsw_rank10_text = f'{unsw_rank10*100:.2f}%' if np.isfinite(unsw_rank10) else 'N/A'

                        print(
                            f'== [Test] Mixed Result: '
                            f'UNSW EER={unsw_eer*100:.2f}% (thr={unsw_thr:.4f}, TAR@0.01%={unsw_tar_1e4*100:.2f}%, Rank-1/5/10={unsw_rank1_text}/{unsw_rank5_text}/{unsw_rank10_text}) | '
                            f'CFPose EER={cfpose_eer*100:.2f}% (thr={cfpose_thr:.4f}, TAR@0.01%={cfpose_tar_1e4*100:.2f}%) | '
                            f'Weighted EER={mixed_weighted_eer*100:.2f}%'
                        )

                        writer.add_scalar('Test/EER_UNSW', unsw_eer, epoch)
                        writer.add_scalar('Test/EER_Threshold_UNSW', unsw_thr, epoch)
                        writer.add_scalar('Test/TAR_at_FAR_0_01pct_UNSW', unsw_tar_1e4, epoch)
                        if np.isfinite(unsw_rank1):
                            writer.add_scalar('Test/Rank1_UNSW', unsw_rank1, epoch)
                        if np.isfinite(unsw_rank5):
                            writer.add_scalar('Test/Rank5_UNSW', unsw_rank5, epoch)
                        if np.isfinite(unsw_rank10):
                            writer.add_scalar('Test/Rank10_UNSW', unsw_rank10, epoch)
                        writer.add_scalar('Test/EER_CFPOSE', cfpose_eer, epoch)
                        writer.add_scalar('Test/EER_Threshold_CFPOSE', cfpose_thr, epoch)
                        writer.add_scalar('Test/TAR_at_FAR_0_01pct_CFPOSE', cfpose_tar_1e4, epoch)
                        writer.add_scalar('Test/EER_Weighted_Mixed', mixed_weighted_eer, epoch)
                        writer.add_scalar('Test/Pairs_UNSW', float(unsw_pairs), epoch)
                        writer.add_scalar('Test/Pairs_CFPOSE', float(cfpose_pairs), epoch)
                        
                        # 使用混合加权 EER 作为 eval_eer
                        epoch_eval_eer = mixed_weighted_eer
                        epoch_eval_threshold = float('nan')  # 混合模式没有单一阈值
                        eval_metric_updated_this_epoch = True
                        last_eval_epoch = epoch + 1
                        last_eval_cache = {
                            'eval_loss': float('nan'),
                            'eval_acc': float('nan'),
                            'eval_eer': float(epoch_eval_eer),
                            'eval_threshold': float(epoch_eval_threshold),
                        }
                        
                        # 保存混合数据集的 EER 结果，用于 checkpoint 命名
                        benchmark_eers = {
                            'unsw': unsw_eer,
                            'cfpose': cfpose_eer
                        }
                else:
                    full_eer, full_eer_threshold, TP, TN, FP, FN, tar_1e4, rank1, rank5, rank10 = test_benchmark(
                        eval_target_model,
                        benchmark_loader,
                        device,
                        dataset_name=args.data_name,
                        epoch_info=epoch+1,
                        ckpt_dir=args.ckpt_dir,
                        use_ntn=ctrl.use_ntn,
                        enable_score_export=False,
                        enable_det_curve=False,
                        enable_score_histogram=True,
                    )
                    if args.data_name == 'unsw' and np.isfinite(rank1):
                        print(
                            f'== [Test] Result: Full-Pool EER = {full_eer*100:.2f}% | '
                            f'Threshold = {full_eer_threshold:.4f} | TAR @ FAR=0.01% = {tar_1e4*100:.2f}% | '
                            f'Rank-1/5/10 = {rank1*100:.2f}%/{rank5*100:.2f}%/{rank10*100:.2f}%'
                        )
                    else:
                        print(f'== [Test] Result: Full-Pool EER = {full_eer*100:.2f}% | Threshold = {full_eer_threshold:.4f} | TAR @ FAR=0.01% = {tar_1e4*100:.2f}%')
                    writer.add_scalar('Test/EER', full_eer, epoch)
                    writer.add_scalar('Test/EER_Threshold', full_eer_threshold, epoch)
                    writer.add_scalar('Test/TAR_at_FAR_0_01pct', tar_1e4, epoch)
                    if args.data_name == 'unsw' and np.isfinite(rank1):
                        writer.add_scalar('Test/Rank1', rank1, epoch)
                        writer.add_scalar('Test/Rank5', rank5, epoch)
                        writer.add_scalar('Test/Rank10', rank10, epoch)
                    
                    # 使用 full-eer 作为 eval_eer
                    epoch_eval_eer = full_eer
                    epoch_eval_threshold = full_eer_threshold
                    eval_metric_updated_this_epoch = True
                    last_eval_epoch = epoch + 1
                    last_eval_cache = {
                        'eval_loss': float('nan'),
                        'eval_acc': float('nan'),
                        'eval_eer': float(epoch_eval_eer),
                        'eval_threshold': float(epoch_eval_threshold),
                    }
            finally:
                if was_training_mode:
                    eval_target_model.train()
                    _enforce_frozen_modules_eval(eval_target_model)
            
            # 评估完成后再次释放内存，降低内存占用
            _release_process_memory(do_cuda_cache_clear=True)
            if args.log_ram_usage:
                writer.add_scalar('System/RAM_RSS_MB/AfterPeriodicEval', _get_rss_mb(), epoch)
            
            # 记录 NTN 模块在测试时的状态
            if ctrl.use_ntn:
                with torch.no_grad():
                    writer.add_scalar('NTN/test_fusion_alpha', torch.sigmoid(eval_target_model.scorer.alpha).item(), epoch)
            
            # 保存最佳检查点
            if eval_metric_updated_this_epoch:
                # 构建包含当前状态的 payload
                payload = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': int(epoch + 1),
                    'global_step': int(global_step),
                    'phase_manager_state': phase_manager.state_dict(),
                    'dynamic_pair_tail_threshold': float(dynamic_pair_tail_threshold),
                    'dynamic_pair_weight_logit': float(pair_weight_logit.detach().item()) if pair_weight_logit is not None else None,

                    'rng_state': _capture_rng_state(),
                    'args': vars(args),
                    'control': ctrl.to_dict(),
                    'metrics': {
                        'train_loss': float(epoch_train_loss),
                        'train_acc': float(epoch_train_acc),
                        'validation_eer': float(epoch_eval_eer),
                    },
                }
                # 保存最佳检查点
                if args.data_name == 'mixed' and 'benchmark_eers' in locals():
                    # 混合模式下根据是否启用only_cfpose_test选择不同的指标
                    if args.only_cfpose_test:
                        # 只使用CFPose的EER作为评估指标
                        ckpt_mgr.maybe_save_best(payload, float(epoch_eval_eer), int(epoch + 1), mixed_eers=benchmark_eers, tar_1e4=float('nan'))
                    else:
                        # 正常模式下使用UNSW的TAR@FAR=0.01%作为指标
                        ckpt_mgr.maybe_save_best(payload, float(epoch_eval_eer), int(epoch + 1), mixed_eers=benchmark_eers, tar_1e4=unsw_tar_1e4)
                else:
                    # 单数据集模式下传递TAR@FAR=0.01%
                    ckpt_mgr.maybe_save_best(payload, float(epoch_eval_eer), int(epoch + 1), tar_1e4=tar_1e4)
                # 写入 manifest
                ckpt_mgr.write_manifest()
            
            has_shown_benchmark_this_run = True

            # 早停机制检查
            if args.early_stop_patience > 0:
                current_val_eer = epoch_eval_eer
                
                # 更新历史EER记录
                val_eer_history.append(current_val_eer)
                if len(val_eer_history) > moving_average_window:
                    val_eer_history.pop(0)
                
                # 计算移动平均EER
                if len(val_eer_history) >= moving_average_window:
                    smoothed_val_eer = sum(val_eer_history) / len(val_eer_history)
                else:
                    smoothed_val_eer = current_val_eer
                
                # 增加验证次数计数器
                validation_count += 1
                
                # 检查是否处于冷却期
                if validation_count <= early_stop_cooling_period:
                    # 冷却期内，只更新最佳值，不增加计数器
                    if smoothed_val_eer < best_val_eer:
                        best_val_eer = smoothed_val_eer
                        print(f'[EarlyStop] Cooling period: Validation EER improved to {smoothed_val_eer*100:.2f}% (smoothed), updating best value.')
                    else:
                        print(f'[EarlyStop] Cooling period: Validation EER did not improve, counter not increased.')
                else:
                    # 冷却期后，正常执行早停逻辑
                    # 检查是否有显著改进
                    if smoothed_val_eer < best_val_eer - early_stop_min_improvement:
                        # 验证指标有显著改进
                        best_val_eer = smoothed_val_eer
                        early_stop_counter = 0
                        print(f'[EarlyStop] Validation EER improved to {smoothed_val_eer*100:.2f}% (smoothed), resetting counter.')
                    elif smoothed_val_eer > best_val_eer + early_stop_tolerance:
                        # 验证指标显著恶化，增加计数器
                        early_stop_counter += 1
                        print(f'[EarlyStop] Validation EER did not improve, counter: {early_stop_counter}/{early_stop_patience}')
                        if early_stop_counter >= early_stop_patience:
                            print(f'[EarlyStop] Patience exhausted, stopping training early after {epoch+1} epochs.')
                            early_stop_triggered = True
                    else:
                        # 验证指标波动在容忍度内，不增加计数器
                        print(f'[EarlyStop] Validation EER fluctuated within tolerance, counter remains: {early_stop_counter}/{early_stop_patience}')

        # ---- 最终测试集评估 (Final Benchmark Evaluation) ----
        # 逻辑：在最后一轮或早停时执行测试集评估，读取 best 模型
        if is_last_epoch or early_stop_triggered:
            if early_stop_triggered:
                print(f'[EarlyStop] Early stop triggered after {epoch+1} epochs, best EER={best_val_eer*100:.2f}%. Start benchmark evaluation')
            if ckpt_mgr.best:
                best_ckpt_info = ckpt_mgr.best[0]
                best_ckpt_path = best_ckpt_info['path']
                best_epoch = best_ckpt_info['epoch']
                
                print(f'== [Benchmark] Loading best checkpoint from: {best_ckpt_path} (Epoch {best_epoch})')
                
                # 释放内存，为加载最佳模型做准备
                _release_process_memory(do_cuda_cache_clear=True)
                if args.log_ram_usage:
                    writer.add_scalar('System/RAM_RSS_MB/BeforeFinalBenchmark', _get_rss_mb(), epoch)
                
                # Load the best checkpoint
                checkpoint = torch.load(best_ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
                
                # Create a new model instance with the same configuration
                best_model = FingerprintNet(
                    feat_dim=128,
                    num_layers=ctrl.num_layers,
                    knn_k=ctrl.knn_k,
                    topk_pool=ctrl.topk_pool,
                    attn_dropout=ctrl.attn_dropout,
                    fps_alpha=ctrl.fps_alpha,
                    tensor_neurons=ctrl.tensor_neurons,
                    dropout_rate=ctrl.dropout_rate,
                    gnn_dropout=ctrl.gnn_dropout,
                    drop_edge_prob=ctrl.drop_edge_prob,
                    feature_jitter=ctrl.feature_jitter,
                    ntn_temperature=ctrl.ntn_temperature,
                    l2_temperature=ctrl.l2_temperature,
                    use_ntn=ctrl.use_ntn,
                    disable_topology_score=ctrl.disable_topology_score,
                    graph_backbone=ctrl.graph_backbone,
                    disable_contraction=ctrl.disable_contraction,
                )
                
                # Load the model state dict
                best_model.load_state_dict(checkpoint['model'])
                
                # Move model to device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                best_model = best_model.to(device)
                
                eval_target_model = best_model
                eval_label = f"Final Benchmark (Best Epoch {best_epoch})"
            else:
                # Fallback to current model if no best checkpoint
                eval_target_model = model
                eval_label = f"Final Benchmark (Epoch {epoch+1})"

            print(f'== [Benchmark] Final Full EER Evaluation on {eval_label} ...')
            
            # 在测试前显式释放训练产生的缓存和计算图，避免测试期 OOM
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Ensure benchmark evaluation does not leak eval mode into the training model.
            was_training_mode = eval_target_model.training
            try:
                if args.data_name == 'mixed':
                    if args.only_cfpose_test:
                        print('== [Benchmark] Mixed mode: only evaluating CFPose (skipping UNSW to save time)...')
                        
                        # 只执行 CFPose 测试
                        cfpose_eer, cfpose_thr, cfpose_tp, cfpose_tn, cfpose_fp, cfpose_fn, cfpose_tar_1e4, _rank1, _rank5, _rank10 = test_benchmark(
                            eval_target_model,
                            benchmark_loader,
                            device,
                            dataset_name='cfpose',
                            epoch_info=epoch+1,
                            ckpt_dir=args.ckpt_dir,
                            use_ntn=ctrl.use_ntn,
                            enable_score_export=False,
                            enable_det_curve=False,
                            enable_score_histogram=True,
                        )
                        
                        cfpose_pairs = cfpose_tp + cfpose_tn + cfpose_fp + cfpose_fn
                        
                        print(
                            f'== [Benchmark] Final Mixed Result: '
                            f'CFPose EER={cfpose_eer*100:.2f}% (thr={cfpose_thr:.4f}, TAR@0.01%={cfpose_tar_1e4*100:.2f}%)'
                        )
                        
                        writer.add_scalar('Benchmark/Final_EER_CFPOSE', cfpose_eer, epoch)
                        writer.add_scalar('Benchmark/Final_EER_Threshold_CFPOSE', cfpose_thr, epoch)
                        writer.add_scalar('Benchmark/Final_TAR_at_FAR_0_01pct_CFPOSE', cfpose_tar_1e4, epoch)
                    else:
                        print('== [Benchmark] Mixed mode: evaluating UNSW and CFPose separately...')

                        unsw_eer, unsw_thr, unsw_tp, unsw_tn, unsw_fp, unsw_fn, unsw_tar_1e4, unsw_rank1, unsw_rank5, unsw_rank10 = test_benchmark(
                            eval_target_model,
                            benchmark_loader,
                            device,
                            dataset_name='unsw',
                            epoch_info=epoch+1,
                            ckpt_dir=args.ckpt_dir,
                            use_ntn=ctrl.use_ntn,
                            enable_score_export=False,
                            enable_det_curve=False,
                            enable_score_histogram=True,
                        )
                        cfpose_eer, cfpose_thr, cfpose_tp, cfpose_tn, cfpose_fp, cfpose_fn, cfpose_tar_1e4, _rank1, _rank5, _rank10 = test_benchmark(
                            eval_target_model,
                            benchmark_loader,
                            device,
                            dataset_name='cfpose',
                            epoch_info=epoch+1,
                            ckpt_dir=args.ckpt_dir,
                            use_ntn=ctrl.use_ntn,
                            enable_score_export=False,
                            enable_det_curve=False,
                            enable_score_histogram=True,
                        )

                        unsw_pairs = unsw_tp + unsw_tn + unsw_fp + unsw_fn
                        cfpose_pairs = cfpose_tp + cfpose_tn + cfpose_fp + cfpose_fn
                        total_pairs = unsw_pairs + cfpose_pairs

                        mixed_weighted_eer = float('nan')
                        if total_pairs > 0 and np.isfinite(unsw_eer) and np.isfinite(cfpose_eer):
                            mixed_weighted_eer = (unsw_eer * unsw_pairs + cfpose_eer * cfpose_pairs) / total_pairs
                        unsw_rank1_text = f'{unsw_rank1*100:.2f}%' if np.isfinite(unsw_rank1) else 'N/A'
                        unsw_rank5_text = f'{unsw_rank5*100:.2f}%' if np.isfinite(unsw_rank5) else 'N/A'
                        unsw_rank10_text = f'{unsw_rank10*100:.2f}%' if np.isfinite(unsw_rank10) else 'N/A'

                        print(
                            f'== [Benchmark] Final Mixed Result: '
                            f'UNSW EER={unsw_eer*100:.2f}% (thr={unsw_thr:.4f}, TAR@0.01%={unsw_tar_1e4*100:.2f}%, Rank-1/5/10={unsw_rank1_text}/{unsw_rank5_text}/{unsw_rank10_text}) | '
                            f'CFPose EER={cfpose_eer*100:.2f}% (thr={cfpose_thr:.4f}, TAR@0.01%={cfpose_tar_1e4*100:.2f}%) | '
                            f'Weighted EER={mixed_weighted_eer*100:.2f}%'
                        )

                        writer.add_scalar('Benchmark/Final_EER_UNSW', unsw_eer, epoch)
                        writer.add_scalar('Benchmark/Final_EER_Threshold_UNSW', unsw_thr, epoch)
                        writer.add_scalar('Benchmark/Final_TAR_at_FAR_0_01pct_UNSW', unsw_tar_1e4, epoch)
                        if np.isfinite(unsw_rank1):
                            writer.add_scalar('Benchmark/Final_Rank1_UNSW', unsw_rank1, epoch)
                        if np.isfinite(unsw_rank5):
                            writer.add_scalar('Benchmark/Final_Rank5_UNSW', unsw_rank5, epoch)
                        if np.isfinite(unsw_rank10):
                            writer.add_scalar('Benchmark/Final_Rank10_UNSW', unsw_rank10, epoch)
                        writer.add_scalar('Benchmark/Final_EER_CFPOSE', cfpose_eer, epoch)
                        writer.add_scalar('Benchmark/Final_EER_Threshold_CFPOSE', cfpose_thr, epoch)
                        writer.add_scalar('Benchmark/Final_TAR_at_FAR_0_01pct_CFPOSE', cfpose_tar_1e4, epoch)
                        writer.add_scalar('Benchmark/Final_EER_Weighted_Mixed', mixed_weighted_eer, epoch)
                else:
                    full_eer, full_eer_threshold, TP, TN, FP, FN, tar_1e4, rank1, rank5, rank10 = test_benchmark(
                        eval_target_model,
                        benchmark_loader,
                        device,
                        dataset_name=args.data_name,
                        epoch_info=epoch+1,
                        ckpt_dir=args.ckpt_dir,
                        use_ntn=ctrl.use_ntn,
                        enable_score_export=False,
                        enable_det_curve=False,
                        enable_score_histogram=True,
                    )
                    if args.data_name == 'unsw' and np.isfinite(rank1):
                        print(
                            f'== [Benchmark] Final Result: Full-Pool EER = {full_eer*100:.2f}% | '
                            f'Threshold = {full_eer_threshold:.4f} | TAR @ FAR=0.01% = {tar_1e4*100:.2f}% | '
                            f'Rank-1/5/10 = {rank1*100:.2f}%/{rank5*100:.2f}%/{rank10*100:.2f}%'
                        )
                    else:
                        print(f'== [Benchmark] Final Result: Full-Pool EER = {full_eer*100:.2f}% | Threshold = {full_eer_threshold:.4f} | TAR @ FAR=0.01% = {tar_1e4*100:.2f}%')
                    writer.add_scalar('Benchmark/Final_EER', full_eer, epoch)
                    writer.add_scalar('Benchmark/Final_EER_Threshold', full_eer_threshold, epoch)
                    writer.add_scalar('Benchmark/Final_TAR_at_FAR_0_01pct', tar_1e4, epoch)
                    if args.data_name == 'unsw' and np.isfinite(rank1):
                        writer.add_scalar('Benchmark/Final_Rank1', rank1, epoch)
                        writer.add_scalar('Benchmark/Final_Rank5', rank5, epoch)
                        writer.add_scalar('Benchmark/Final_Rank10', rank10, epoch)
            finally:
                if was_training_mode:
                    eval_target_model.train()
                    _enforce_frozen_modules_eval(eval_target_model)
            
            # 最终测试评估完成后释放内存
            _release_process_memory(do_cuda_cache_clear=True)
            if args.log_ram_usage:
                writer.add_scalar('System/RAM_RSS_MB/AfterFinalBenchmark', _get_rss_mb(), epoch)
            
            # 如果是早停触发的评估，执行完基准测试后需要跳出循环
            if early_stop_triggered:
                break

    # 关闭 TensorBoard writer
    writer.close()
    print(f'\nTensorBoard logs saved to: {log_dir}')
    
    print(f'[CKPT] Directory: {args.ckpt_dir} (best.pt, last.pt, manifest.json)')

    # Optional final copy to a user-specified path (does not affect pruning)
    if args.save_path:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dynamic_pair_tail_threshold': float(dynamic_pair_tail_threshold),
                'dynamic_pair_weight_logit': float(pair_weight_logit.detach().item()) if pair_weight_logit is not None else None,

                'rng_state': _capture_rng_state(),
                'args': vars(args),
                'control': ctrl.to_dict(),
            },
            args.save_path,
        )
        print(f'[CKPT] Wrote final copy to: {args.save_path}')


if __name__ == '__main__':
    # default: run training to make sure everything works end-to-end
    train_main()
