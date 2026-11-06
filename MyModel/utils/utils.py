import os
import random
import glob
from typing import Dict, List, Tuple
import numpy as np
import torch


def _parse_unsw_name(file_path: str):
    """Parse UNSW filename: Subject_Finger_Session_View.mat."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parts = stem.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid UNSW filename format: {file_path}")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric UNSW filename format: {file_path}") from exc


def _parse_numeric_stem(file_path: str) -> int:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    try:
        return int(stem)
    except ValueError:
        # Fallback: use first numeric token for stems like "123_4"
        parts = stem.split('_')
        return int(parts[0])


def _parse_cfpose_name(file_path: str):
    """Parse CFPose filename variants: <identity>.mat or <identity>_<frame>.mat."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parts = stem.split('_')
    try:
        identity = int(parts[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid CFPose filename format: {file_path}") from exc

    frame_idx = None
    if len(parts) >= 2:
        try:
            frame_idx = int(parts[1])
        except ValueError:
            frame_idx = None
    # Use -1 as deterministic fallback for files without explicit frame suffix.
    return identity, (-1 if frame_idx is None else frame_idx)


def _infer_data_name_from_path(file_path: str) -> str:
    lower = file_path.lower()
    if 'unsw' in lower:
        return 'unsw'
    if 'cfpose' in lower:
        return 'cfpose'
    return 'cfpose'


def sort_files_for_dataset(files, data_name: str):
    """Sort files with dataset-aware deterministic keys."""
    if data_name == 'unsw':
        files.sort(key=_parse_unsw_name)
    elif data_name == 'cfpose':
        files.sort(key=_parse_cfpose_name)
    elif data_name == 'mixed':
        # Keep deterministic cross-domain order and preserve per-domain natural keys.
        def _mixed_key(fp):
            dn = _infer_data_name_from_path(fp)
            if dn == 'unsw':
                return (1, _parse_unsw_name(fp))
            return (0, _parse_cfpose_name(fp))
        files.sort(key=_mixed_key)
    else:
        files.sort(key=_parse_numeric_stem)


def _get_unsw_subject(file_path: str) -> int:
    subject, _, _, _ = _parse_unsw_name(file_path)
    return subject


def extract_subject_from_unsw_finger_id(finger_ids: torch.Tensor) -> torch.Tensor:
    """UNSW finger id encoding: 1_000_000 + subject*1000 + finger."""
    return (finger_ids.long() - 1000000) // 1000


def compute_gf_score_from_H(
    H: torch.Tensor,
    remove_self_loop: bool = True,
    use_log: bool = True,
):
    if H.dim() != 3:
        raise ValueError(f'Expected H to be 3D (B,N,*) but got shape={tuple(H.shape)}')
    
    # A: [B, N, N] - 邻接矩阵
    A = torch.bmm(H, H.transpose(1, 2))
    if remove_self_loop:
        B, N, _ = A.shape
        # 移除自环
        eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        A = A * (1.0 - eye)
        
    # d: [B, N] - 度向量 (每个节点的连接强度)
    d = A.sum(dim=2)
    
    # Ad: [B, N] - 邻居度数的加权和
    # 公式：Ad = A × d，表示每个节点邻居的平均连通性
    Ad = torch.bmm(A, d.unsqueeze(2)).squeeze(2)
    
    # structural_score: [B, N] - 拉普拉斯显著性
    # 核心思想：如果一个节点的度数远高于其邻居，则该节点是重要的
    score_batch = d * d - Ad
    
    # 按每个样本独立进行 MinMax 归一化到 [0, 1]，抑制噪声影响
    s_min = score_batch.min(dim=1, keepdim=True)[0]
    s_max = score_batch.max(dim=1, keepdim=True)[0]
    score_batch = (score_batch - s_min) / (s_max - s_min + 1e-12)
    
    if use_log:
        # 使用 Log1p 进一步平滑分布，使其更适合作为深度学习的 Logits
        # 乘以 5.0 增加对比度，使高分点更突出
        score_batch = torch.log1p(score_batch * 5.0)
        
    return score_batch


def get_finger_id(file_name: str, data_name: str) -> int:
    effective_data_name = data_name
    if data_name == 'mixed':
        effective_data_name = _infer_data_name_from_path(file_name)

    base = os.path.basename(file_name).replace('.mat', '')
    parts = base.split('_')
    
    # 为不同数据集设置 ID 偏移量，避免冲突
    id_offset = 0
    if effective_data_name == 'unsw':
        id_offset = 1000000  # UNSW 的 ID 从 1,000,000 开始
    # cfpose 保持从 0 开始
    
    if effective_data_name == 'unsw':
        # unsw rule: [SubjectID, FingerID, Session, View]
        try:
            subject_id = int(parts[0])
            finger_id = int(parts[1])
            
            # 放弃视角区分，同一手指不同视角分配同一 ID
            base_id = subject_id * 1000 + finger_id
            return base_id + id_offset
        except (IndexError, ValueError):
            return -1
    else:
        # cfpose rule (Default)
        try:
            identity, _ = _parse_cfpose_name(file_name)
            base_id = int(identity)
            return base_id + id_offset
        except (IndexError, ValueError):
            return -1


def count_model_parameters(model: torch.nn.Module, verbose: bool = True):
    trainable_params = 0
    frozen_params = 0
    trainable_details = {}
    frozen_details = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable_params += num_params
            # 按模块层级统计
            module_name = name.split('.')[0]
            if module_name not in trainable_details:
                trainable_details[module_name] = 0
            trainable_details[module_name] += num_params
        else:
            frozen_params += num_params
            # 按模块层级统计
            module_name = name.split('.')[0]
            if module_name not in frozen_details:
                frozen_details[module_name] = 0
            frozen_details[module_name] += num_params
    
    total_params = trainable_params + frozen_params
    
    if verbose:
        print("\n" + "="*70)
        print("Model Parameter Statistics")
        print("="*70)
        print(f"  Total Parameters:    {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  Trainable:          {trainable_params:,} ({trainable_params / 1e6:.2f}M, {trainable_params/total_params*100:.1f}%)")
        print(f"  Frozen:             {frozen_params:,} ({frozen_params / 1e6:.2f}M, {frozen_params/total_params*100:.1f}%)")
        print("-"*70)
        
        if trainable_details:
            print("  Trainable by Module:")
            for module, count in sorted(trainable_details.items(), key=lambda x: x[1], reverse=True):
                print(f"    {module:20s}: {count:,} ({count/1e6:.2f}M, {count/trainable_params*100:.1f}%)")
        
        if frozen_details:
            print("-"*70)
            print("  Frozen by Module:")
            for module, count in sorted(frozen_details.items(), key=lambda x: x[1], reverse=True):
                print(f"    {module:20s}: {count:,} ({count/1e6:.2f}M, {count/frozen_params*100:.1f}%)")
        
        print("="*70 + "\n")
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_details': trainable_details,
        'frozen_details': frozen_details
    }


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if ckpt_path is None or ckpt_path == '':
        return None
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    # Legacy artifacts include Python/NumPy training metadata. Use trusted files.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint: {ckpt_path}')
    if len(missing) > 0:
        print(f'[Warn] Missing keys ({len(missing)}): {missing[:8]}{" ..." if len(missing) > 8 else ""}')
    if len(unexpected) > 0:
        print(f'[Warn] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{" ..." if len(unexpected) > 8 else ""}')
    return ckpt if isinstance(ckpt, dict) else None


def collate_list(batch):
    # 检查所有样本的节点数是否一致
    num_nodes = [len(sample['coords']) for sample in batch]
    if len(set(num_nodes)) > 1:
        raise ValueError(f"Node numbers inconsistent in batch: {num_nodes}. "
                        f"All samples must have the same number of nodes.")
    
    return {
        'full_img': torch.stack([sample['full_img'] for sample in batch]),
        'coords': torch.stack([sample['coords'] for sample in batch]),
        'node_mask': torch.stack([sample['node_mask'] for sample in batch]),
        'W': torch.stack([sample['W'] for sample in batch]),
        'file_name': [sample['file_name'] for sample in batch],
        'file_path': [sample['file_path'] for sample in batch],
        'id': torch.tensor([sample['id'] for sample in batch]),
    }

def collate_triplet_list(batch):
    anchors = [sample['anchor'] for sample in batch]
    positives = [sample['positive'] for sample in batch]
    negatives = [sample['negative'] for sample in batch]
    
    return {
        'anchor': collate_list(anchors),
        'positive': collate_list(positives),
        'negative': collate_list(negatives),
    }

def split_files_fixed_uwa(files):
    # Test: subject 1..100, Val: subject 101..110, Train: subject 111..150
    if len(files) != 8958:
        print(f"[Warn] Total files ({len(files)}) does not match UWA paper count (8958). Split might be inaccurate.")

    train_files = []
    val_files = []
    test_files = []
    out_of_range_files = []

    for fp in files:
        try:
            subject = _get_unsw_subject(fp)
        except ValueError:
            out_of_range_files.append(fp)
            continue

        if 1 <= subject <= 100:
            test_files.append(fp)
        elif 101 <= subject <= 110:
            val_files.append(fp)
        elif 111 <= subject <= 150:
            train_files.append(fp)
        else:
            out_of_range_files.append(fp)

    sort_files_for_dataset(test_files, 'unsw')
    sort_files_for_dataset(val_files, 'unsw')
    sort_files_for_dataset(train_files, 'unsw')

    test_subjects = sorted({_get_unsw_subject(fp) for fp in test_files})
    val_subjects = sorted({_get_unsw_subject(fp) for fp in val_files})
    train_subjects = sorted({_get_unsw_subject(fp) for fp in train_files})
    
    # 检查是否有subject泄露
    overlap_test_val = sorted(set(test_subjects).intersection(val_subjects))
    overlap_test_train = sorted(set(test_subjects).intersection(train_subjects))
    overlap_val_train = sorted(set(val_subjects).intersection(train_subjects))
    
    if overlap_test_val:
        raise RuntimeError(f"[UWA Split Error] Test-Val subject leakage detected: {overlap_test_val[:10]}")
    if overlap_test_train:
        raise RuntimeError(f"[UWA Split Error] Test-Train subject leakage detected: {overlap_test_train[:10]}")
    if overlap_val_train:
        raise RuntimeError(f"[UWA Split Error] Val-Train subject leakage detected: {overlap_val_train[:10]}")

    if len(test_subjects) != 100:
        print(f"[Warn] Expected 100 test subjects, got {len(test_subjects)}")
    if len(val_subjects) != 10:
        print(f"[Warn] Expected 10 validation subjects, got {len(val_subjects)}")
    if len(train_subjects) != 40:
        print(f"[Warn] Expected 40 train subjects, got {len(train_subjects)}")
    if out_of_range_files:
        print(f"[Warn] Ignored {len(out_of_range_files)} files out of protocol range or parse failure.")

    print(
        f"UWA split by subject: Train={len(train_files)} (subjects {min(train_subjects) if train_subjects else 'NA'}-{max(train_subjects) if train_subjects else 'NA'}), "
        f"Val={len(val_files)} (subjects {min(val_subjects) if val_subjects else 'NA'}-{max(val_subjects) if val_subjects else 'NA'}), "
        f"Test={len(test_files)} (subjects {min(test_subjects) if test_subjects else 'NA'}-{max(test_subjects) if test_subjects else 'NA'})"
    )
    
    test_ids = list(set([get_finger_id(f, 'unsw') for f in test_files]))
    
    return train_files, val_files, test_files, sorted(test_ids)


def split_files_cfpose(files, test_count=1200, val_count=20, train_count=180):
    # CFPose 划分：按有序文件列表前缀取测试，再取验证，其余作为训练。
    total = len(files)
    test_count = int(test_count)
    val_count = int(val_count)
    train_count = int(train_count)

    if test_count < 0 or val_count < 0 or train_count < 0:
        raise ValueError(
            f"Invalid CFPose split counts: test={test_count}, val={val_count}, train={train_count}"
        )

    expected_total = test_count + val_count + train_count
    if expected_total != total:
        print(
            f"[Warn] CFPose split sum mismatch: total={total}, "
            f"test+val+train={expected_total}. Auto-adjust train_count to fit total."
        )
        if test_count + val_count > total:
            test_count = min(test_count, total)
            val_count = min(val_count, max(0, total - test_count))
        train_count = max(0, total - test_count - val_count)

    test_split_point = min(test_count, total)
    val_split_point = min(test_split_point + val_count, total)

    test_files = files[:test_split_point]
    val_files = files[test_split_point:val_split_point]
    train_files = files[val_split_point:]

    # 提取测试集 ID 用于日志
    test_ids = list(set([get_finger_id(f, 'cfpose') for f in test_files]))

    print(
        "CFPose 数据划分："
        f"Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}, "
        f"cfg(test/val/train)={test_count}/{val_count}/{train_count}, Test IDs={len(test_ids)}"
    )

    return train_files, val_files, test_files, sorted(test_ids)


def split_files_mixed_protocol(
    cfpose_files,
    unsw_files,
    seed=42,
    cfpose_test_count=1200,
    cfpose_val_count=20,
    cfpose_train_count=180,
):
    """Mixed protocol split: UNSW always uses fixed UWA split; CFPose split mode is configurable."""
    c_train, c_val, c_test, c_test_ids = split_files_cfpose(
        cfpose_files,
        test_count=cfpose_test_count,
        val_count=cfpose_val_count,
        train_count=cfpose_train_count,
    )

    u_train, u_val, u_test, u_test_ids = split_files_fixed_uwa(unsw_files)

    train_files = list(c_train) + list(u_train)
    val_files = list(c_val) + list(u_val)
    test_files = list(c_test) + list(u_test)
    sort_files_for_dataset(train_files, 'mixed')
    sort_files_for_dataset(val_files, 'mixed')
    sort_files_for_dataset(test_files, 'mixed')
    test_ids = sorted(set(c_test_ids).union(set(u_test_ids)))
    print(
        f"Mixed fixed protocol split: Train={len(train_files)} (cfpose={len(c_train)}, unsw={len(u_train)}), "
        f"Val={len(val_files)} (cfpose={len(c_val)}, unsw={len(u_val)}), "
        f"Test={len(test_files)} (cfpose={len(c_test)}, unsw={len(u_test)})"
    )
    return train_files, val_files, test_files, test_ids

def _split_files_by_id(files, test_ratio=0.2, seed=0, data_name='cfpose'):
    """Split file paths by identity (id parsed based on data_name rule)."""
    id_to_files = {}
    for fp in files:
        file_data_name = _infer_data_name_from_path(fp) if data_name == 'mixed' else data_name
        sample_id = get_finger_id(fp, file_data_name)
        if sample_id < 0:
            continue
        id_to_files.setdefault(sample_id, []).append(fp)

    ids = sorted(id_to_files.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)

    num_test_ids = max(1, int(round(len(ids) * test_ratio)))
    test_ids = set(ids[:num_test_ids])

    train_files = []
    test_files = []
    for sample_id, fps in id_to_files.items():
        if sample_id in test_ids:
            test_files.extend(fps)
        else:
            train_files.extend(fps)

    # 使用数据集感知排序，保持与主流程一致
    sort_files_for_dataset(train_files, data_name)
    sort_files_for_dataset(test_files, data_name)
    return train_files, test_files, sorted(list(test_ids))


def collect_all_files(data_name: str, data_config: Dict[str, str]):
    """Collect files for single or mixed datasets with deterministic ordering."""
    if data_name == 'mixed':
        cfpose_dir = data_config.get('cfpose')
        unsw_dir = data_config.get('unsw')
        if (not cfpose_dir) or (not os.path.exists(cfpose_dir)):
            raise RuntimeError(f"CFPose data dir not found for mixed mode: {cfpose_dir}")
        if (not unsw_dir) or (not os.path.exists(unsw_dir)):
            raise RuntimeError(f"UNSW data dir not found for mixed mode: {unsw_dir}")

        cfpose_files = glob.glob(os.path.join(cfpose_dir, '*.mat'))
        unsw_files = glob.glob(os.path.join(unsw_dir, '*.mat'))
        sort_files_for_dataset(cfpose_files, 'cfpose')
        sort_files_for_dataset(unsw_files, 'unsw')

        all_files = list(cfpose_files) + list(unsw_files)
        sort_files_for_dataset(all_files, 'mixed')
        return all_files, cfpose_files, unsw_files

    data_dir = data_config.get(data_name)
    if not data_dir or not os.path.exists(data_dir):
        raise RuntimeError(f"data_dir for '{data_name}' not found: {data_dir}")
    all_files = glob.glob(os.path.join(data_dir, '*.mat'))
    sort_files_for_dataset(all_files, data_name)
    return all_files, [], []


def filter_files_by_dataset(files, dataset_name):
    """根据数据集名称筛选文件"""
    filtered = []
    for f in files:
        if dataset_name == 'cfpose' and 'cfpose' in f.lower():
            filtered.append(f)
        elif dataset_name == 'unsw' and 'unsw' in f.lower():
            filtered.append(f)
    return filtered

def compute_pairwise_similarity(model, z1, z2, use_ntn=True):
    """统一的相似度路由：默认 NTN；关闭后退化为纯 L2 相似度。"""
    if use_ntn and hasattr(model, 'scorer'):
        return model.scorer(z1, z2)
    l2_score = torch.norm(z1 - z2, dim=-1)
    # 当use_ntn为False时，使用默认的l2_temperature
    l2_temperature = 1.0
    return torch.exp(-(l2_score / l2_temperature))


def move_sample_to_device(sample, device):
    # 如果是 triplet 格式（包含 anchor/positive/negative 键）
    if 'anchor' in sample and 'positive' in sample and 'negative' in sample:
        return {
            'anchor': move_sample_to_device(sample['anchor'], device),
            'positive': move_sample_to_device(sample['positive'], device),
            'negative': move_sample_to_device(sample['negative'], device),
        }
    # 否则是单个样本格式
    return {
        'full_img': sample['full_img'].to(device),
        'coords': sample['coords'].to(device),
        'node_mask': sample['node_mask'].to(device),
        'W': sample['W'].to(device),  # 预计算的邻接矩阵
        'file_name': sample.get('file_name', ''),
        'file_path': sample.get('file_path', ''),
        'id': sample.get('id', -1),
    }
