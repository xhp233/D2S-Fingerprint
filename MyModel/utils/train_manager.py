import math
from dataclasses import dataclass, asdict
from typing import Any, Dict
import torch


class SimplePhaseManager:
    """简化的阶段管理器，只负责跟踪阶段状态和处理固定的阶段切换。"""
    def __init__(self, warm_up_epochs, batch_all_epochs, mixed_hard_epochs):
        self.max_epochs = [warm_up_epochs, batch_all_epochs, mixed_hard_epochs, float('inf')]
        self.current_phase = 0  # 0: warm_up, 1: batch_all, 2: mixed_hard, 3: standard
        self.phase_epoch = 0
        self.transition_from_phase = 0

    def state_dict(self):
        return {
            'current_phase': self.current_phase,
            'phase_epoch': self.phase_epoch,
            'transition_from_phase': self.transition_from_phase,
        }

    def load_state_dict(self, state):
        self.current_phase = state.get('current_phase', 0)
        self.phase_epoch = state.get('phase_epoch', 0)
        self.transition_from_phase = state.get('transition_from_phase', max(0, self.current_phase - 1))

    def _switch_phase(self, target_phase, reason):
        target_phase = int(max(0, min(3, target_phase)))
        if target_phase == self.current_phase:
            return

        old_phase = self.current_phase
        direction = 'Forward' if target_phase > old_phase else 'Backward'
        print(f"-> [Phase Switch][{direction}] Jumping from Phase {old_phase} to {target_phase}. Reason: {reason}.")

        self.transition_from_phase = old_phase
        self.current_phase = target_phase
        self.phase_epoch = 0
        self.last_transition = {
            'from': old_phase,
            'to': target_phase,
            'direction': direction,
            'reason': reason,
        }


@dataclass
class TrainingControlManager:
    lr: float
    weight_decay: float
    margin: float
    num_layers: int
    knn_k: int
    tensor_neurons: int
    batch_size: int
    topk_pool: int
    fps_alpha: float
    attn_dropout: float
    dropout_rate: float
    gnn_dropout: float
    drop_edge_prob: float
    feature_jitter: float
    disable_topology_score: bool
    graph_backbone: str
    disable_contraction: bool
    scorer_weight_decay_scale: float
    ntn_temperature: float
    l2_temperature: float
    enable_pair_loss: bool
    pair_loss_weight: float
    pair_tail_threshold: float
    pair_tail_alpha: float
    pair_tail_tau: float
    enable_dynamic_pair_tail_threshold: bool
    dynamic_pair_tail_quantile: float
    dynamic_pair_tail_min: float
    dynamic_pair_tail_max: float
    dynamic_pair_tail_momentum: float
    dynamic_pair_tail_max_samples: int
    enable_dynamic_pair_weight: bool
    dynamic_pair_weight_pre_hard: float
    dynamic_pair_weight_init: float
    dynamic_pair_weight_max: float
    dynamic_pair_weight_lr_scale: float
    enable_mixed_dynamic_domain_weight: bool
    mixed_domain_weight_init_log_var: float
    mixed_domain_weight_lr_scale: float
    augment: bool
    use_ntn: bool

    @classmethod
    def from_args(cls, args):
        return cls(
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            margin=float(args.margin),
            num_layers=int(args.num_layers),
            knn_k=int(args.knn_k),
            tensor_neurons=int(args.tensor_neurons),
            batch_size=int(args.batch_size),
            topk_pool=int(args.topk_pool),
            fps_alpha=float(args.fps_alpha),
            attn_dropout=float(args.attn_dropout),
            dropout_rate=float(args.dropout_rate),
            gnn_dropout=float(args.gnn_dropout),
            drop_edge_prob=float(args.drop_edge_prob),
            feature_jitter=float(args.feature_jitter),
            disable_topology_score=bool(args.disable_topology_score),
            graph_backbone=str(args.graph_backbone),
            disable_contraction=bool(getattr(args, 'disable_contraction', False)),
            scorer_weight_decay_scale=float(args.scorer_weight_decay_scale),
            ntn_temperature=float(args.ntn_temperature),
            l2_temperature=float(args.l2_temperature),
            enable_pair_loss=bool(args.enable_pair_loss),
            pair_loss_weight=float(args.pair_loss_weight),
            pair_tail_threshold=float(args.pair_tail_threshold),
            pair_tail_alpha=float(args.pair_tail_alpha),
            pair_tail_tau=float(args.pair_tail_tau),
            enable_dynamic_pair_tail_threshold=bool(args.enable_dynamic_pair_tail_threshold),
            dynamic_pair_tail_quantile=float(args.dynamic_pair_tail_quantile),
            dynamic_pair_tail_min=float(args.dynamic_pair_tail_min),
            dynamic_pair_tail_max=float(args.dynamic_pair_tail_max),
            dynamic_pair_tail_momentum=float(args.dynamic_pair_tail_momentum),
            dynamic_pair_tail_max_samples=int(args.dynamic_pair_tail_max_samples),
            enable_dynamic_pair_weight=bool(args.enable_dynamic_pair_weight),
            dynamic_pair_weight_pre_hard=float(args.dynamic_pair_weight_pre_hard),
            dynamic_pair_weight_init=float(args.dynamic_pair_weight_init),
            dynamic_pair_weight_max=float(args.dynamic_pair_weight_max),
            dynamic_pair_weight_lr_scale=float(args.dynamic_pair_weight_lr_scale),
            enable_mixed_dynamic_domain_weight=bool(args.enable_mixed_dynamic_domain_weight),
            mixed_domain_weight_init_log_var=float(args.mixed_domain_weight_init_log_var),
            mixed_domain_weight_lr_scale=float(args.mixed_domain_weight_lr_scale),
            augment=bool(args.augment),
            use_ntn=bool(args.use_ntn),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_optimizer(model, ctrl: TrainingControlManager):
    # Respect upstream freeze settings (e.g. freeze_cnn).
    # Do not forcefully unfreeze parameters here.

    cnn_params = []
    gnn_params = []
    fusion_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder.backbone' in name:
            cnn_params.append(param)
        elif 'fusion_w' in name or 'scorer' in name:
            fusion_params.append(param)
        else:
            gnn_params.append(param)

    param_groups = []
    if len(gnn_params) > 0:
        param_groups.append({'params': gnn_params, 'lr': ctrl.lr})
    if len(cnn_params) > 0:
        param_groups.append({'params': cnn_params, 'lr': ctrl.lr})
    if len(fusion_params) > 0:
        param_groups.append({
            'params': fusion_params,
            'lr': ctrl.lr,
            'weight_decay': ctrl.weight_decay * ctrl.scorer_weight_decay_scale,
        })

    if len(param_groups) == 0:
        raise RuntimeError('No optimizer param groups created. Please check freeze settings.')

    return torch.optim.AdamW(param_groups, weight_decay=ctrl.weight_decay)


def assert_resume_consistency(curr_args, ckpt_args):
    """Strictly verify key training/data settings before resuming from checkpoint."""
    if not isinstance(ckpt_args, dict):
        raise RuntimeError('[Resumer] Checkpoint does not contain valid args dict; aborting in strict mode.')

    critical_keys = [
        'batch_size', 'eval_batch_size',
        'warm_up_epochs', 'batch_all_epochs', 'mixed_hard_epochs',
        'pair_loss_weight', 'pair_tail_threshold', 'pair_tail_alpha', 'pair_tail_tau',
        'enable_dynamic_pair_tail_threshold', 'dynamic_pair_tail_quantile',
        'dynamic_pair_tail_min', 'dynamic_pair_tail_max',
        'dynamic_pair_tail_momentum', 'dynamic_pair_tail_max_samples',
        'enable_dynamic_pair_weight', 'dynamic_pair_weight_pre_hard',
        'dynamic_pair_weight_init', 'dynamic_pair_weight_max', 'dynamic_pair_weight_lr_scale',
        'enable_feature_constancy', 'feature_constancy_weight', 'feature_constancy_ckpt',
        'enable_mixed_dynamic_domain_weight', 'mixed_domain_weight_init_log_var', 'mixed_domain_weight_lr_scale',
        'lr', 'weight_decay', 'margin',
        'topk_pool',
        'num_layers', 'knn_k', 'tensor_neurons',
        'dropout_rate', 'attn_dropout', 'gnn_dropout', 'fps_alpha',
        'drop_edge_prob', 'feature_jitter',
        'disable_topology_score', 'graph_backbone', 'disable_contraction',
        'scorer_weight_decay_scale', 'ntn_temperature', 'l2_temperature',
        'use_ntn', 'augment',
        'augment_profile', 'micro_aug_max_rotate', 'micro_aug_max_translate_ratio',
        'freeze_cnn', 'peft_metric_tuning', 'hgnn_train_last_n_layers',
        'mixed_static_weight_unsw', 'mixed_static_weight_cfpose',
        'mixed_unsw_sampling_ratio',
        'cfpose_test_count', 'cfpose_val_count', 'cfpose_train_count',
    ]

    mismatches = []
    for key in critical_keys:
        if (not hasattr(curr_args, key)) or (key not in ckpt_args):
            continue

        curr_val = getattr(curr_args, key)
        ckpt_val = ckpt_args[key]

        same = False
        if isinstance(curr_val, float) or isinstance(ckpt_val, float):
            try:
                same = abs(float(curr_val) - float(ckpt_val)) <= 1e-12
            except Exception:
                same = str(curr_val) == str(ckpt_val)
        else:
            same = curr_val == ckpt_val

        if not same:
            mismatches.append((key, ckpt_val, curr_val))

    if mismatches:
        msg_lines = [
            '[Resumer] Strict consistency check failed. Resume aborted.',
            '[Resumer] Mismatched settings (checkpoint -> current):'
        ]
        for key, ckpt_v, curr_v in mismatches:
            msg_lines.append(f'  - {key}: {ckpt_v} -> {curr_v}')
        raise RuntimeError('\n'.join(msg_lines))

