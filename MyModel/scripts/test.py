import os
import sys
import gc
import time


import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from MyModel.utils.utils import extract_subject_from_unsw_finger_id
from MyModel.utils.utils import compute_pairwise_similarity, move_sample_to_device, load_checkpoint_into_model
import pandas as pd


def visualize_topk_distribution(img_tensor, coords, selected_indices, scores, save_path):
    """
    img_tensor: [3, H, W] or [1, H, W]
    coords: [N, 2] (归一化坐标 -1 到 1)
    selected_indices: [K] (按选择顺序排列的索引)
    scores: [N] (所有节点的得分)
    """
    if img_tensor.dim() == 3:
        img = img_tensor[0].cpu().numpy()
    else:
        img = img_tensor.cpu().numpy()

    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-6)

    H, W = img.shape

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')

    # 获取选中点的得分
    selected_scores = scores[selected_indices].cpu().numpy()
    
    # 归一化得分到 [0, 1] 范围
    if len(selected_scores) > 0:
        score_min = selected_scores.min()
        score_max = selected_scores.max()
        if score_max > score_min:
            normalized_scores = (selected_scores - score_min) / (score_max - score_min)
        else:
            normalized_scores = np.ones_like(selected_scores)
    else:
        normalized_scores = []

    # 按选择顺序绘制点，使用得分来生成颜色和透明度
    total_points = len(selected_indices)
    for i, (idx, norm_score) in enumerate(zip(selected_indices, normalized_scores)):
        # 颜色：得分越高越红，越低越蓝
        # 使用 RGB 颜色，红到蓝的渐变
        r = norm_score  # 得分高 -> 红色多
        b = 1.0 - norm_score  # 得分低 -> 蓝色多
        g = 0.0  # 不使用绿色
        color = (r, g, b)
        
        # 透明度：得分越高越不透明
        alpha = 0.5 + (norm_score * 0.5)  # 透明度范围 0.5-1.0
        
        # 获取坐标
        x, y = coords[idx].cpu().numpy()
        screen_x = (x + 1) / 2 * W
        screen_y = (y + 1) / 2 * H
        
        # 绘制点
        plt.scatter(screen_x, screen_y, c=[color], s=40, edgecolors='white', linewidths=0.5, marker='o', alpha=alpha)

    plt.title(f"Top-K Distribution (K={total_points}) - Color: Red=High Score, Blue=Low Score", fontsize=15)
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()


def compute_pairwise_similarity_dual(model, f1, f2):
    """
    同时计算 NTN Fused 和 L2-only 两个路由的相似度分数
    """
    if not hasattr(model, 'scorer'):
        l2_scores = F.cosine_similarity(f1, f2)
        return l2_scores, l2_scores

    f1 = F.normalize(f1, p=2, dim=1)
    f2 = F.normalize(f2, p=2, dim=1)

    with torch.no_grad():
        fused_scores = model.scorer(f1, f2)
        l2_scores = F.cosine_similarity(f1, f2)

    return fused_scores, l2_scores


@torch.no_grad()
def test_benchmark(
    model,
    dataloader,
    device,
    dataset_name=None,
    epoch_info=None,
    ckpt_dir=None,
    use_ntn=True,
    report_both_routes=False,
    pair_batch_size=1024,
    enable_visualization=False,
    enable_score_export=False,
    enable_det_curve=False,
    enable_score_histogram=True,
    gallery_strategy='first',
    gallery_metadata=None,
):
    model.eval()
    all_feats = []
    all_ids = []
    all_file_paths = []
    debug_info_for_vis = None

    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating EER {dataset_name or ''}"):
            sample = move_sample_to_device(batch, device)
            result = model(sample, return_debug=True)
            if isinstance(result, tuple):
                feat, logits, debug_info = result
            else:
                feat = result
                debug_info = None

            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            feat = F.normalize(feat, p=2, dim=1)

            all_feats.append(feat.cpu())
            all_ids.append(sample['id'].cpu())
            all_file_paths.extend(sample['file_path'])

            if debug_info is not None and 'selected_idx' in debug_info and 'scores' in debug_info and debug_info_for_vis is None:
                debug_info_for_vis = {
                    'selected_idx': debug_info['selected_idx'],
                    'scores': debug_info['scores'],
                    'full_img': sample.get('full_img', None),
                    'coords': sample.get('coords', None),
                }

    if len(all_feats) == 0:
        return float('nan'), float('nan'), 0, 0, 0, 0, float('nan'), float('nan'), float('nan'), float('nan')

    if debug_info_for_vis is not None and ckpt_dir is not None and enable_visualization:
        vis_dir = os.path.join(ckpt_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        b_idx = 0
        img_to_show = debug_info_for_vis['full_img'][b_idx]
        coords_to_show = debug_info_for_vis['coords'][b_idx]
        selected_indices = debug_info_for_vis['selected_idx'][b_idx]
        scores = debug_info_for_vis['scores'][b_idx]

        fname = f"topk_epoch_{epoch_info or 'eval'}_{dataset_name or 'data'}.png"
        save_path = os.path.join(vis_dir, fname)

        visualize_topk_distribution(img_to_show, coords_to_show, selected_indices, scores, save_path)
        print(f"[Visualization] Saved Top-K distribution map to: {save_path}")
    elif debug_info_for_vis is None and enable_visualization:
        print(f"[Visualization] No 'selected_idx' or 'scores' found in debug_info or ckpt_dir is None. Skipping visualization.")

    feats = torch.cat(all_feats, dim=0)
    ids = torch.cat(all_ids, dim=0)

    # 计算特征向量在每个维度上的标准差
    print("[计算] 特征向量维度标准差...")
    feat_std = torch.std(feats, dim=0)
    print(f"特征维度: {feat_std.size(0)}")
    print(f"各维度标准差: {feat_std.tolist()[:10]}..." if feat_std.size(0) > 10 else f"各维度标准差: {feat_std.tolist()}")
    print(f"平均标准差: {torch.mean(feat_std).item()}")
    print(f"标准差范围: [{torch.min(feat_std).item()}, {torch.max(feat_std).item()}]")
    
    # 执行主成分分析 (PCA)
    print("\n[计算] 主成分分析 (PCA)...")
    feats_np = feats.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feats_np)
    
    # 计算方差解释率
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"前 3 个主成分的方差解释率: {explained_variance_ratio}")
    print(f"累积方差解释率: {cumulative_variance_ratio}")
    print(f"前 3 个主成分总共解释了 {cumulative_variance_ratio[2] * 100:.2f}% 的方差")
    
    # 评估 PCA 结果
    if cumulative_variance_ratio[2] > 0.8:
        print("[警告] 前 3 个主成分解释了 80% 以上的方差，说明模型可能只在学习少数几种特征")
    else:
        print("[信息] 前 3 个主成分解释的方差比例合理，模型学习的特征分布较为均匀")

    if dataset_name is not None:
        mask = []
        for fp in all_file_paths:
            if dataset_name == 'cfpose' and 'cfpose' in fp.lower():
                mask.append(True)
            elif dataset_name == 'unsw' and 'unsw' in fp.lower():
                mask.append(True)
            else:
                mask.append(False)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        if not mask_tensor.any():
            return float('nan'), float('nan'), 0, 0, 0, 0, float('nan'), float('nan'), float('nan'), float('nan')
        feats = feats[mask_tensor]
        ids = ids[mask_tensor]
        all_file_paths = [fp for idx, fp in enumerate(all_file_paths) if mask[idx]]

    n_test = feats.size(0)
    protocol_expected_pairs = None
    protocol_detail = ""

    has_scorer = hasattr(model, 'scorer')
    if report_both_routes and has_scorer:
        route_keys = ['ntn_fused', 'l2_only']
    elif report_both_routes:
        print("[Dual Route] Warning: model.scorer 不存在，仅可评估 L2-only 路由。")
        route_keys = ['l2_only']
    else:
        route_keys = ['primary']

    if report_both_routes and has_scorer:
        primary_route = 'ntn_fused' if use_ntn else 'l2_only'
    elif report_both_routes:
        primary_route = 'l2_only'
    else:
        primary_route = 'primary'

    route_labels = {
        'primary': 'NTN Fused' if use_ntn else 'L2-only',
        'ntn_fused': 'NTN Fused',
        'l2_only': 'L2-only',
    }

    gen_score_chunks_by_route = {k: [] for k in route_keys}
    imp_score_chunks_by_route = {k: [] for k in route_keys}
    rank1_hits_by_route = {k: 0 for k in route_keys}
    rank5_hits_by_route = {k: 0 for k in route_keys}
    rank10_hits_by_route = {k: 0 for k in route_keys}
    rank1_total_by_route = {k: 0 for k in route_keys}
    
    # 添加存储分数和文件路径对的列表
    score_pairs_by_route = {k: [] for k in route_keys}

    feats = feats.to(device)
    ids = ids.to(device)
    batch_size = max(1, int(pair_batch_size))

    def _append_score_chunks(route_key, row_scores, is_gen_mask, query_idx, compare_idxs):
        gen_chunk = row_scores[is_gen_mask]
        if gen_chunk.numel() > 0:
            gen_score_chunks_by_route[route_key].append(gen_chunk.numpy().astype(np.float32, copy=False))
        imp_chunk = row_scores[~is_gen_mask]
        if imp_chunk.numel() > 0:
            imp_score_chunks_by_route[route_key].append(imp_chunk.numpy().astype(np.float32, copy=False))
        
        # 收集分数和文件路径对
        for i, (score, is_gen) in enumerate(zip(row_scores.cpu().numpy(), is_gen_mask.numpy())):
            pair_type = 'genuine' if is_gen else 'impostor'
            score_pairs_by_route[route_key].append({
                'score': score,
                'pair_type': pair_type,
                'query_path': all_file_paths[query_idx],
                'compare_path': all_file_paths[compare_idxs[i]]
            })

    def _compute_score_map(f1_batch, f2_batch):
        if report_both_routes and has_scorer:
            fused_scores, l2_scores = compute_pairwise_similarity_dual(model, f1_batch, f2_batch)
            return {
                'ntn_fused': fused_scores.cpu(),
                'l2_only': l2_scores.cpu(),
            }
        if report_both_routes:
            l2_scores = compute_pairwise_similarity(model, f1_batch, f2_batch, use_ntn=False)
            return {'l2_only': l2_scores.cpu()}

        scores = compute_pairwise_similarity(model, f1_batch, f2_batch, use_ntn=use_ntn)
        return {'primary': scores.cpu()}

    if dataset_name == 'unsw':
        print(f"\n[UWA Protocol] Separating Capture 1 (Reference) and Capture 2 (Query)...")
        is_cap1 = torch.tensor([os.path.basename(fp).split('_')[2] == '1' for fp in all_file_paths], device=device)
        is_cap2 = torch.tensor([os.path.basename(fp).split('_')[2] == '2' for fp in all_file_paths], device=device)

        feats1, ids1 = feats[is_cap1], ids[is_cap1]
        feats2, ids2 = feats[is_cap2], ids[is_cap2]

        subject1 = extract_subject_from_unsw_finger_id(ids1)
        subject2 = extract_subject_from_unsw_finger_id(ids2)

        all_subjects = torch.cat([subject1, subject2])
        min_subject = int(all_subjects.min().item()) if len(all_subjects) > 0 else 1
        max_subject = int(all_subjects.max().item()) if len(all_subjects) > 0 else 150

        valid_mask1 = (subject1 >= min_subject) & (subject1 <= max_subject)
        valid_mask2 = (subject2 >= min_subject) & (subject2 <= max_subject)

        feats1, ids1 = feats1[valid_mask1], ids1[valid_mask1]
        feats2, ids2 = feats2[valid_mask2], ids2[valid_mask2]
        protocol_expected_pairs = int(len(feats1) * len(feats2))
        protocol_detail = f"N_ref x N_query = {len(feats1)} x {len(feats2)}"

        print(f"[UWA Protocol] Reference (1st capture): {len(feats1)}, Query (2nd capture): {len(feats2)}")
        unique_ref_fingers = int(torch.unique(ids1).numel())
        unique_query_fingers = int(torch.unique(ids2).numel())
        print(f"[UWA Protocol] Unique fingers: ref={unique_ref_fingers}, query={unique_query_fingers}")

        with torch.no_grad():
            for i in range(len(feats2)):
                query_feat = feats2[i].unsqueeze(0)
                query_id = ids2[i]

                row_score_chunks = {k: [] for k in route_keys}
                for j in range(0, len(feats1), batch_size):
                    gallery_batch = feats1[j : j + batch_size]
                    q_batch = query_feat.expand(gallery_batch.size(0), -1)

                    score_map = _compute_score_map(q_batch, gallery_batch)
                    for route_key, route_scores in score_map.items():
                        row_score_chunks[route_key].append(route_scores)

                row_scores_by_route = {}
                for route_key in route_keys:
                    if len(row_score_chunks[route_key]) == 0:
                        row_scores_by_route[route_key] = torch.empty((0,), dtype=torch.float32)
                    else:
                        row_scores_by_route[route_key] = torch.cat(row_score_chunks[route_key], dim=0)

                is_gen = (ids1 == query_id).cpu()
                # 计算 compare_idxs：feats1 对应 all_file_paths 的前 len(feats1) 个元素
                compare_idxs = [j for j in range(len(feats1))]
                # 计算 query_idx：feats2 对应 all_file_paths 从 len(feats1) 开始的元素
                query_idx = len(feats1) + i
                for route_key, row_scores in row_scores_by_route.items():
                    _append_score_chunks(route_key, row_scores, is_gen, query_idx, compare_idxs)

                    if row_scores.numel() > 0:
                        top10_scores, top10_indices = torch.topk(row_scores, min(10, len(row_scores)))
                        top10_ids = ids1[top10_indices].tolist()
                        query_id_item = query_id.item()
                        rank1_hits_by_route[route_key] += int(top10_ids[0] == query_id_item)
                        rank5_hits_by_route[route_key] += int(query_id_item in top10_ids[:5])
                        rank10_hits_by_route[route_key] += int(query_id_item in top10_ids)
                        rank1_total_by_route[route_key] += 1

    else:
        protocol_expected_pairs = int(n_test * (n_test - 1) // 2)
        protocol_detail = f"C(N,2) = {n_test} x {n_test - 1} / 2"
        with torch.no_grad():
            for i in range(n_test - 1):
                target_feat = feats[i].unsqueeze(0)
                target_id = ids[i]

                compare_feats = feats[i+1:]
                compare_ids = ids[i+1:]
                n_compare = compare_feats.size(0)

                row_score_chunks = {k: [] for k in route_keys}
                for j in range(0, n_compare, batch_size):
                    f1_batch = target_feat.expand(min(batch_size, n_compare - j), -1)
                    f2_batch = compare_feats[j : j + batch_size]

                    score_map = _compute_score_map(f1_batch, f2_batch)
                    for route_key, route_scores in score_map.items():
                        row_score_chunks[route_key].append(route_scores)

                row_scores_by_route = {}
                for route_key in route_keys:
                    if len(row_score_chunks[route_key]) == 0:
                        row_scores_by_route[route_key] = torch.empty((0,), dtype=torch.float32)
                    else:
                        row_scores_by_route[route_key] = torch.cat(row_score_chunks[route_key], dim=0)

                is_gen = (compare_ids == target_id).cpu()
                # 计算 compare_idxs：从 i+1 开始的索引
                compare_idxs = [j for j in range(i+1, n_test)]
                for route_key, row_scores in row_scores_by_route.items():
                    _append_score_chunks(route_key, row_scores, is_gen, i, compare_idxs)

        if dataset_name == 'cfpose':
            def _parse_cfpose_view_idx(file_path):
                stem = os.path.splitext(os.path.basename(file_path))[0]
                parts = stem.split('_')
                if len(parts) < 2:
                    return -1
                try:
                    return int(parts[1])
                except ValueError:
                    return -1

            view_idx = torch.tensor([_parse_cfpose_view_idx(fp) for fp in all_file_paths], device=device)
            invalid_view_count = int((view_idx < 1).sum().item())
            if invalid_view_count > 0:
                print(f"[CFPose Rank-1] Warning: {invalid_view_count} samples have invalid view index and will be ignored.")

            is_gallery = (view_idx == 1)
            is_query = (view_idx >= 2) & (view_idx <= 10)

            gallery_indices_all = torch.nonzero(is_gallery, as_tuple=False).squeeze(1).tolist()
            query_indices_all = torch.nonzero(is_query, as_tuple=False).squeeze(1).tolist()

            if gallery_strategy == 'auto' and ckpt_dir is not None:
                route_key_for_gallery = 'ntn' if use_ntn else 'l2'
                gallery_json_path_ntn = os.path.join(ckpt_dir, 'rank_n_ntn.json')
                gallery_json_path_l2 = os.path.join(ckpt_dir, 'rank_n_l2.json')
                gallery_json_path = gallery_json_path_ntn if use_ntn else gallery_json_path_l2
                
                if os.path.exists(gallery_json_path):
                    import json as json_lib
                    with open(gallery_json_path, 'r') as f:
                        best_gallery_local_idx = json_lib.load(f)
                    print(f"[CFPose Rank-1] Auto gallery strategy: loading {gallery_json_path}")
                    
                    finger_to_global_idx = {}
                    for global_idx in range(len(ids)):
                        fid = int(ids[global_idx].item())
                        if fid not in finger_to_global_idx:
                            finger_to_global_idx[fid] = []
                        finger_to_global_idx[fid].append(global_idx)
                    
                    gallery_indices = []
                    for fid_str, local_idx in best_gallery_local_idx.items():
                        fid = int(fid_str)
                        if fid in finger_to_global_idx and local_idx < len(finger_to_global_idx[fid]):
                            gallery_indices.append(finger_to_global_idx[fid][local_idx])
                    gallery_indices = sorted(gallery_indices)
                    print(f"[CFPose Rank-1] Auto gallery: selected {len(gallery_indices)} images")
                else:
                    print(f"[CFPose Rank-1] Warning: {gallery_json_path} not found, falling back to first-as-gallery")
                    gallery_index_by_id = {}
                    for idx in gallery_indices_all:
                        fid = int(ids[idx].item())
                        if fid not in gallery_index_by_id:
                            gallery_index_by_id[fid] = idx
                    gallery_indices = sorted(gallery_index_by_id.values())
            elif gallery_strategy == 'fixed3':
                finger_to_global_idx = {}
                for global_idx in range(len(ids)):
                    fid = int(ids[global_idx].item())
                    if fid not in finger_to_global_idx:
                        finger_to_global_idx[fid] = []
                    finger_to_global_idx[fid].append(global_idx)
                
                gallery_indices = []
                for fid in sorted(finger_to_global_idx.keys()):
                    indices = finger_to_global_idx[fid]
                    if len(indices) > 3:
                        gallery_indices.append(indices[3])
                    elif len(indices) > 0:
                        gallery_indices.append(indices[0])
                gallery_indices = sorted(gallery_indices)
                print(f"[CFPose Rank-1] Fixed-Index-3 strategy: selected {len(gallery_indices)} images")
            elif gallery_strategy in ('angle_gt', 'angle_jia'):
                # Objective "most-frontal" gallery: per finger, pick the view with the
                # smallest |in-plane refine angle|. GT file is (140,10,3) -> component 1;
                # Jia-predicted file is (140,10). Angles used ONLY for the eval protocol.
                from scipy.io import loadmat as _loadmat
                _ar_path = gallery_metadata or os.path.join('data', 'cfpose', 'angle_refine_cfpose_gt.mat' if gallery_strategy == 'angle_gt' else 'my_angle_refine_all.mat')
                if not os.path.isfile(_ar_path):
                    raise FileNotFoundError('Gallery metadata not found: '+str(_ar_path))
                _ar = _loadmat(_ar_path)['angle_refine']
                _ang = _ar[:, :, 1].astype(float) if _ar.ndim == 3 else _ar.astype(float)
                finger_to_global_idx = {}
                for global_idx in range(len(ids)):
                    fid = int(ids[global_idx].item())
                    finger_to_global_idx.setdefault(fid, []).append(global_idx)
                gallery_indices = []
                for fid in sorted(finger_to_global_idx.keys()):
                    best_idx, best_abs = None, None
                    for gi in finger_to_global_idx[fid]:
                        v = int(view_idx[gi].item())  # frame index 1..10
                        if not (1 <= v <= 10 and 1 <= fid <= _ang.shape[0]):
                            continue
                        a = abs(float(_ang[fid - 1, v - 1]))
                        if best_abs is None or a < best_abs:
                            best_abs, best_idx = a, gi
                    if best_idx is not None:
                        gallery_indices.append(best_idx)
                gallery_indices = sorted(gallery_indices)
                _gset = set(gallery_indices)
                # probe set = all valid views EXCEPT each finger's gallery view
                query_indices_all = [i for i in range(len(ids))
                                     if 1 <= int(view_idx[i].item()) <= 10 and i not in _gset]
                print(f"[CFPose Rank-1] Angle-frontal gallery ({gallery_strategy}): "
                      f"selected {len(gallery_indices)} galleries (min |in-plane angle|)")
            else:
                gallery_index_by_id = {}
                for idx in gallery_indices_all:
                    fid = int(ids[idx].item())
                    if fid not in gallery_index_by_id:
                        gallery_index_by_id[fid] = idx
                gallery_indices = sorted(gallery_index_by_id.values())

            if len(gallery_indices) == 0 or len(query_indices_all) == 0:
                print("[CFPose Rank-1] Warning: Empty gallery/query under first-as-gallery protocol.")
            else:
                gallery_feats = feats[gallery_indices]
                gallery_ids = ids[gallery_indices]
                gallery_id_set = set(int(v) for v in gallery_ids.cpu().tolist())
                query_indices = [idx for idx in query_indices_all if int(ids[idx].item()) in gallery_id_set]

                strategy_name = {
                    'auto': 'Auto-Gallery',
                    'fixed3': 'Fixed-Index-3',
                    'first': 'First-as-Gallery',
                    'angle_gt': 'Angle-Frontal-GT',
                    'angle_jia': 'Angle-Frontal-JiaPred',
                }.get(gallery_strategy, 'First-as-Gallery')
                print(
                    f"[CFPose Rank-1] {strategy_name} protocol: "
                    f"gallery={len(gallery_indices)}, query(excluding gallery)={len(query_indices)}"
                )

                with torch.no_grad():
                    for idx in query_indices:
                        query_feat = feats[idx].unsqueeze(0)
                        query_id = ids[idx]

                        row_score_chunks = {k: [] for k in route_keys}
                        for j in range(0, len(gallery_feats), batch_size):
                            gallery_batch = gallery_feats[j : j + batch_size]
                            q_batch = query_feat.expand(gallery_batch.size(0), -1)
                            score_map = _compute_score_map(q_batch, gallery_batch)
                            for route_key, route_scores in score_map.items():
                                row_score_chunks[route_key].append(route_scores)

                        row_scores_by_route = {}
                        for route_key in route_keys:
                            if len(row_score_chunks[route_key]) == 0:
                                row_scores_by_route[route_key] = torch.empty((0,), dtype=torch.float32)
                            else:
                                row_scores_by_route[route_key] = torch.cat(row_score_chunks[route_key], dim=0)

                        for route_key, row_scores in row_scores_by_route.items():
                            if row_scores.numel() > 0:
                                top10_scores, top10_indices = torch.topk(row_scores, min(10, len(row_scores)))
                                top10_ids = gallery_ids[top10_indices].tolist()
                                query_id_item = query_id.item()
                                rank1_hits_by_route[route_key] += int(top10_ids[0] == query_id_item)
                                rank5_hits_by_route[route_key] += int(query_id_item in top10_ids[:5])
                                rank10_hits_by_route[route_key] += int(query_id_item in top10_ids)
                                rank1_total_by_route[route_key] += 1

    def _evaluate_single_route(route_key):
        route_label = route_labels.get(route_key, route_key)
        rank1_total = rank1_total_by_route.get(route_key, 0)
        rank1_acc = (
            rank1_hits_by_route.get(route_key, 0) / rank1_total
            if rank1_total > 0
            else float('nan')
        )
        rank5_acc = (
            rank5_hits_by_route.get(route_key, 0) / rank1_total
            if rank1_total > 0
            else float('nan')
        )
        rank10_acc = (
            rank10_hits_by_route.get(route_key, 0) / rank1_total
            if rank1_total > 0
            else float('nan')
        )

        def _merge_chunks(chunks):
            if len(chunks) == 0:
                return np.empty((0,), dtype=np.float32)
            if len(chunks) == 1:
                return chunks[0]
            return np.concatenate(chunks, axis=0)

        gen_scores = _merge_chunks(gen_score_chunks_by_route[route_key])
        imp_scores = _merge_chunks(imp_score_chunks_by_route[route_key])
        print(
            f"\n[Matching Validated][{route_label}] "
            f"Genuine Pairs: {len(gen_scores)}, Impostor Pairs: {len(imp_scores)}"
        )

        def _q(values, q):
            if len(values) == 0:
                return float('nan')
            return float(np.quantile(values, q))

        high_score_cut = 0.50
        imp_high_ratio = float(np.mean(imp_scores >= high_score_cut)) if len(imp_scores) > 0 else float('nan')
        print(
            f"[Score Dist][{route_label}] "
            f"gen(p50/p90/p99)={_q(gen_scores, 0.50):.4f}/{_q(gen_scores, 0.90):.4f}/{_q(gen_scores, 0.99):.4f}, "
            f"imp(p50/p90/p99)={_q(imp_scores, 0.50):.4f}/{_q(imp_scores, 0.90):.4f}/{_q(imp_scores, 0.99):.4f}, "
            f"imp@>={high_score_cut:.2f}={imp_high_ratio:.4%}"
        )

        if len(gen_scores) == 0 or len(imp_scores) == 0:
            return float('nan'), float('nan'), 0, 0, 0, 0, float('nan'), rank1_acc, rank5_acc, rank10_acc

        y_true = np.concatenate([np.ones_like(gen_scores), np.zeros_like(imp_scores)])
        y_scores = np.concatenate([gen_scores, imp_scores])

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr

        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        def get_tpr_at_fpr(target_fpr):
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                best_idx = idx[-1]
                return tpr[best_idx], thresholds[best_idx]
            return 0.0, float('nan')

        tpr_at_1e3, thresh_1e3 = get_tpr_at_fpr(1e-3)
        tpr_at_1e4, thresh_1e4 = get_tpr_at_fpr(1e-4)

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_curve, precision_curve)

        eps = 1e-10
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + eps)
        max_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_idx]
        opt_f1_threshold = pr_thresholds[max_f1_idx] if max_f1_idx < len(pr_thresholds) else pr_thresholds[-1]

        TP = int(np.sum(gen_scores >= eer_threshold))
        FN = int(np.sum(gen_scores < eer_threshold))
        FP = int(np.sum(imp_scores >= eer_threshold))
        TN = int(np.sum(imp_scores < eer_threshold))

        total_pairs = TP + TN + FP + FN
        match_type_label = "UWA Bipartite Protocol" if dataset_name == 'unsw' else "Upper Triangle Only"
        print("\n" + "="*70)
        print(f"Classification Stats ({match_type_label}) [{route_label}]")
        print("="*70)
        if protocol_detail:
            print(f"  Pair construction:   {protocol_detail}")
        if protocol_expected_pairs is not None:
            print(f"  Expected pairs:      {protocol_expected_pairs:8d}")
        print("--- 1. Stats Based on EER Threshold ---")
        print(f"  Total pairs:         {total_pairs:8d}")
        print(f"  Positive pairs:      {TP+FN:8d}")
        print(f"  Negative pairs:      {TN+FP:8d}")
        print(f"  EER threshold:       {eer_threshold:.6f}")
        print(f"  EER:                 {eer:.6f} ({eer*100:.4f}%)")
        print(f"  EER TP: {TP:8d} | EER TN: {TN:8d}")
        print(f"  EER FP: {FP:8d} | EER FN: {FN:8d}")
        if np.isfinite(imp_high_ratio):
            print(f"  Impostor>={high_score_cut:.2f}:      {imp_high_ratio:.4%}")
        if TP + FP > 0:
            print(f"  EER Precision:       {TP / (TP + FP):.6f}")
        if TP + FN > 0:
            print(f"  EER F1 Score:        {(2 * TP / (2*TP + FP + FN)) if (TP+FP+FN)>0 else 0.0:.6f}")
        if protocol_expected_pairs is not None and total_pairs != protocol_expected_pairs:
            print(f"  [Warn] Pair count mismatch: expected={protocol_expected_pairs}, actual={total_pairs}")

        print("\n--- 2. High Requirement (Low FAR) Stats ---")
        print(f"  TAR @ FAR=0.1%  (1e-3): {tpr_at_1e3:.6f} (Threshold: {thresh_1e3:.6f})")
        print(f"  TAR @ FAR=0.01% (1e-4): {tpr_at_1e4:.6f} (Threshold: {thresh_1e4:.6f})")
        print(f"  TAR @ FAR=0.00% ( 0  ): {tpr[0]:.6f} (Threshold: {thresholds[0]:.6f})")
        if np.isfinite(rank1_acc):
            print(f"  Rank-1 Accuracy:     {rank1_acc:.6f} ({rank1_acc*100:.2f}%)")
        if np.isfinite(rank5_acc):
            print(f"  Rank-5 Accuracy:     {rank5_acc:.6f} ({rank5_acc*100:.2f}%)")
        if np.isfinite(rank10_acc):
            print(f"  Rank-10 Accuracy:    {rank10_acc:.6f} ({rank10_acc*100:.2f}%)")

        print("\n--- 3. Precision-Recall Stats ---")
        print(f"  PR-AUC:              {pr_auc:.6f}")
        print(f"  Max F1 Score:        {max_f1:.6f} (Threshold: {opt_f1_threshold:.6f})")
        print("="*70 + "\n")

        # 导出分数和文件路径对为 CSV
        if enable_score_export and ckpt_dir is not None and len(score_pairs_by_route.get(route_key, [])) > 0:
            export_dir = os.path.join(ckpt_dir, 'score_analysis')
            os.makedirs(export_dir, exist_ok=True)
            
            # 创建 DataFrame
            df = pd.DataFrame(score_pairs_by_route[route_key])
            
            # 保存为 CSV
            csv_filename = f"score_pairs_{route_key}_{dataset_name or 'all'}.csv"
            csv_path = os.path.join(export_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"[Score Export] Saved score pairs to: {csv_path}")

        # 绘制 DET 曲线（默认关闭）
        if ckpt_dir is not None and enable_det_curve:
            # 创建保存目录
            viz_dir = os.path.join(ckpt_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 计算 FMR (False Match Rate) 和 FNMR (False Non-Match Rate)
            fmr = fpr  # FMR = FPR
            fnmr = fnr  # FNMR = FNR
            
            # 创建 DET 曲线
            plt.figure(figsize=(10, 8))
            
            # 绘制 DET 曲线
            plt.plot(fmr, fnmr, 'b-', linewidth=2, label=f'DET Curve ({route_label})')
            
            # 绘制 EER 点
            plt.plot(eer, eer, 'ro', markersize=8, label=f'EER = {eer*100:.2f}%')
            
            # 绘制参考线 (EER 线)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='EER Reference')
            
            # 设置坐标轴为指数形式，右上为 0 点
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim([1e-6, 1])
            plt.ylim([1e-6, 1])
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            
            # 设置坐标轴标签和标题
            plt.xlabel('False Match Rate (FMR)')
            plt.ylabel('False Non-Match Rate (FNMR)')
            plt.title(f'DET Curve - {route_label} ({dataset_name or "All"})')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # 保存图像
            det_filename = f"det_curve_{route_key}_{dataset_name or 'all'}.png"
            det_path = os.path.join(viz_dir, det_filename)
            plt.savefig(det_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Saved DET curve to: {det_path}")
        
        # 绘制匹配分数分布直方图（默认开启）
        if ckpt_dir is not None and enable_score_histogram:
            # 创建保存目录
            viz_dir = os.path.join(ckpt_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            print(f"[Visualization] Drawing score distribution histogram...")
            plt.figure(figsize=(12, 6))
            
            # 设置柱子宽度为 0.01
            bins = np.arange(0, 1.01, 0.01)
            
            # 绘制 Genuine 分数分布
            plt.hist(gen_scores, bins=bins, alpha=0.6, label='Genuine', edgecolor='black', color='blue')
            
            # 绘制 Impostor 分数分布
            plt.hist(imp_scores, bins=bins, alpha=0.6, label='Impostor', edgecolor='black', color='red')
            
            # 设置纵轴为指数形式
            plt.yscale('log')
            
            # 设置坐标轴标签和标题
            plt.xlabel('Matching Score')
            plt.ylabel('Frequency (log scale)')
            plt.title(f'Score Distribution - {route_label} ({dataset_name or "All"})')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # 保存图像（添加 epoch 信息防止覆盖）
            if epoch_info is not None:
                hist_filename = f"score_distribution_epoch{epoch_info}_{route_key}_{dataset_name or 'all'}.png"
            else:
                hist_filename = f"score_distribution_{route_key}_{dataset_name or 'all'}.png"
            hist_path = os.path.join(viz_dir, hist_filename)
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Saved score distribution histogram to: {hist_path}")

        return eer, eer_threshold, TP, TN, FP, FN, tpr_at_1e4, rank1_acc, rank5_acc, rank10_acc

    route_results = {}
    for route_key in route_keys:
        route_results[route_key] = _evaluate_single_route(route_key)
        gen_score_chunks_by_route[route_key].clear()
        imp_score_chunks_by_route[route_key].clear()
        gc.collect()

    if report_both_routes and has_scorer and {'ntn_fused', 'l2_only'}.issubset(route_results.keys()):
        ntn_eer, _, _, _, _, _, ntn_tar_1e4, ntn_rank1, ntn_rank5, ntn_rank10 = route_results['ntn_fused']
        l2_eer, _, _, _, _, _, l2_tar_1e4, l2_rank1, l2_rank5, l2_rank10 = route_results['l2_only']
        print("[Dual Route Summary] "
              f"NTN Fused: EER={ntn_eer*100:.4f}%, TAR@1e-4={ntn_tar_1e4:.6f}, Rank-1/5/10={ntn_rank1:.6f}/{ntn_rank5:.6f}/{ntn_rank10:.6f} | "
              f"L2-only: EER={l2_eer*100:.4f}%, TAR@1e-4={l2_tar_1e4:.6f}, Rank-1/5/10={l2_rank1:.6f}/{l2_rank5:.6f}/{l2_rank10:.6f}")

    result = route_results[primary_route]

    del all_feats
    del all_ids
    del all_file_paths
    del feats
    del ids
    gc.collect()

    return result


# =============================================================================
# 计时测试函数 - 比较 NTN 和 L2-only 的推理时间
# =============================================================================

@torch.no_grad()
def benchmark_inference_time(model, device, num_iterations=1000, batch_size=64, feat_dim=128):
    """
    比较 NTN Fused 和 L2-only 两条路径的推理时间
    
    Args:
        model: 已加载的模型
        device: 计算设备
        num_iterations: 迭代次数（取平均）
        batch_size: 每次迭代的 batch size
        feat_dim: 特征维度
    """
    model.eval()
    
    print("\n" + "="*70)
    print("推理时间对比测试")
    print("="*70)
    print(f"迭代次数: {num_iterations}")
    print(f"Batch Size: {batch_size}")
    print(f"特征维度: {feat_dim}")
    
    has_scorer = hasattr(model, 'scorer')
    if not has_scorer:
        print("[警告] 模型没有 scorer，只能测试 L2-only 路径")
        return
    
    # 生成随机特征作为测试数据
    f1 = F.normalize(torch.randn(num_iterations, batch_size, feat_dim, device=device), p=2, dim=-1)
    f2 = F.normalize(torch.randn(num_iterations, batch_size, feat_dim, device=device), p=2, dim=-1)
    
    # 预热运行（确保 CUDA 初始化）
    print("\n[预热] 运行 10 次预热迭代...")
    for i in range(10):
        _ = model.scorer(f1[i], f2[i])
        _ = F.cosine_similarity(f1[i], f2[i])
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # 测试 NTN Fused 路径
    print("\n[测试 NTN Fused]...")
    start_time = time.time()
    for i in range(num_iterations):
        _ = model.scorer(f1[i], f2[i])
    torch.cuda.synchronize() if device.type == 'cuda' else None
    ntn_time = time.time() - start_time
    
    # 测试 L2-only 路径
    print("[测试 L2-only]...")
    start_time = time.time()
    for i in range(num_iterations):
        _ = F.cosine_similarity(f1[i], f2[i])
    torch.cuda.synchronize() if device.type == 'cuda' else None
    l2_time = time.time() - start_time
    
    # 计算统计信息
    total_pairs = num_iterations * batch_size
    ntn_avg_time = ntn_time / num_iterations * 1000  # ms per batch
    l2_avg_time = l2_time / num_iterations * 1000    # ms per batch
    ntn_per_pair = ntn_time / total_pairs * 1000000  # us per pair
    l2_per_pair = l2_time / total_pairs * 1000000    # us per pair
    
    print("\n" + "="*70)
    print("推理时间对比结果")
    print("="*70)
    print(f"总处理样本对: {total_pairs:,}")
    print("-"*70)
    print(f"NTN Fused:")
    print(f"  总时间: {ntn_time:.4f} 秒")
    print(f"  平均每次 Batch: {ntn_avg_time:.4f} ms")
    print(f"  平均每样本对: {ntn_per_pair:.4f} us")
    print("-"*70)
    print(f"L2-only:")
    print(f"  总时间: {l2_time:.4f} 秒")
    print(f"  平均每次 Batch: {l2_avg_time:.4f} ms")
    print(f"  平均每样本对: {l2_per_pair:.4f} us")
    print("-"*70)
    print(f"NTN 比 L2 慢: {(ntn_time / l2_time - 1) * 100:.2f}%")
    print(f"NTN 耗时是 L2 的: {ntn_time / l2_time:.2f} 倍")
    print("="*70)
    
    return ntn_time, l2_time


# =============================================================================
# 独立运行 Benchmark 测试
# =============================================================================

def main():
    import argparse
    import torch.utils.data as data_cls
    from MyModel.model import FingerprintNet
    from MyModel.datasets import FingerprintDataset
    from MyModel.utils.utils import collect_all_files, split_files_fixed_uwa, split_files_cfpose, split_files_mixed_protocol

    from MyModel.utils.cli import data_paths, parse_with_config

    parser = argparse.ArgumentParser(description='指纹识别模型 Benchmark 测试工具')

    parser.add_argument('--data_name', type=str, default=None, choices=['cfpose', 'unsw', 'mixed'],
                       help='数据集名称（默认从检查点自动读取）')
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='要加载的模型检查点路径')
    parser.add_argument('--cfpose_test_count', type=int, default=1200,
                       help='CFPose 划分中测试集样本数（按有序文件前缀切分）')
    parser.add_argument('--cfpose_val_count', type=int, default=20,
                       help='CFPose 划分中验证集样本数（紧随测试集）')
    parser.add_argument('--cfpose_train_count', type=int, default=180,
                       help='CFPose 划分中训练集样本数（其余部分）')

    parser.add_argument('--num_layers', type=int, default=None,
                       help='超图神经网络的层数（默认从检查点读取）')
    parser.add_argument('--knn_k', type=int, default=None,
                       help='KNN 构建超图时的邻居数量 k（默认从检查点读取）')
    parser.add_argument('--topk_pool', type=int, default=None,
                       help='Top-K 池化：每个图保留的节点数量（默认从检查点读取）')
    parser.add_argument('--tensor_neurons', type=int, default=None,
                       help='Tensor Network 中隐藏层的神经元数量（默认从检查点读取）')
    parser.add_argument('--dropout_rate', type=float, default=None,
                       help='通用 dropout 概率（默认从检查点读取）')
    parser.add_argument('--gnn_dropout', type=float, default=None,
                       help='GNN 输出层的 dropout 概率（默认从检查点读取）')
    parser.add_argument('--drop_edge_prob', type=float, default=None,
                       help='超图边的随机丢弃概率（默认从检查点读取）')
    parser.add_argument('--feature_jitter', type=float, default=None,
                       help='特征随机抖动的方差（默认从检查点读取）')
    parser.add_argument('--fps_alpha', type=float, default=None,
                       help='加权 FPS 中距离的权重（默认从检查点读取）')
    parser.add_argument('--attn_dropout', type=float, default=None,
                       help='注意力分数的 dropout 概率（默认从检查点读取）')
    parser.add_argument('--ntn_temperature', type=float, default=None,
                       help='NTN 分支温度缩放（默认从检查点读取）')
    parser.add_argument('--l2_temperature', type=float, default=None,
                       help='L2 分支温度缩放（默认从检查点读取）')
    use_ntn_group = parser.add_mutually_exclusive_group()
    use_ntn_group.add_argument('--use_ntn', dest='eval_use_ntn', action='store_true',
                       help='测试时启用 NTN 打分器（默认跟随检查点）')
    use_ntn_group.add_argument('--no-use_ntn', '--no-use-ntn', dest='eval_use_ntn', action='store_false',
                       help='测试时禁用 NTN 打分器（允许加载 use_ntn 训练得到的模型）')
    parser.set_defaults(eval_use_ntn=None)
    parser.add_argument('--disable_topology_score', action='store_true', default=None,
                       help='禁用拓扑分数融合（默认从检查点读取）')
    parser.add_argument('--graph_backbone', type=str, default=None, choices=['hgnn', 'gcn'],
                       help='图骨干类型（默认从检查点读取）')

    parser.add_argument('--batch_size', type=int, default=16,
                       help='测试的 batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载的 worker 数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    parser.add_argument('--report_both_routes', action='store_true',
                       help='单次评测同时输出 NTN Fused 与 L2-only 两套路由结果（共享同一批特征与配对前向）')
    parser.set_defaults(report_both_routes=True)
    parser.add_argument('--enable_visualization', action='store_true',
                       help='启用可视化功能（包括 Top-K 分布）')
    parser.set_defaults(enable_visualization=False)
    parser.add_argument('--enable_det_curve', action='store_true',
                       help='启用 DET 曲线绘制（默认关闭）')
    parser.set_defaults(enable_det_curve=False)
    parser.add_argument('--enable_score_histogram', action='store_true',
                       help='启用分数分布直方图绘制（默认开启）')
    parser.set_defaults(enable_score_histogram=True)
    parser.add_argument('--enable_score_export', action='store_true',
                       help='启用分数导出功能（保存 score_analysis 文件夹和 CSV 文件）')
    parser.set_defaults(enable_score_export=False)
    parser.add_argument('--gallery_strategy', type=str, default='first', choices=['first', 'auto', 'fixed3', 'angle_gt', 'angle_jia'],
                       help='CFPose gallery 选择策略: first=View1, auto=内部比对最优, fixed3=第4张图(index=3), '
                            'angle_gt=按 CFPose GT refine 角选最正视(min|in-plane|), angle_jia=按 Jia 预测角选最正视')
    parser.set_defaults(gallery_strategy='first')

    parser.add_argument('--data_root', default=os.environ.get('HGNN_DATA_ROOT', 'data'), help='Root containing cfpose/ and unsw/ dataset folders')
    parser.add_argument('--config', default=None, help='Explicit JSON configuration; command-line flags override it')
    parser.add_argument('--gallery_metadata', default=None, help='Local angle_refine MAT used only by angle_gt/angle_jia identification')

    args = parse_with_config(parser)
    DATA_CONFIG = data_paths(args.data_root)

    print("="*70)
    print("指纹识别模型 Benchmark 测试")
    print("="*70)
    print(f"检查点: {args.ckpt_path}")
    print(f"设备: {args.device}")
    print("="*70)

    if not os.path.exists(args.ckpt_path):
        print(f"[错误] 检查点文件不存在: {args.ckpt_path}")
        sys.exit(1)

    print(f"\n[1/4] 加载检查点...")
    checkpoint = torch.load(args.ckpt_path, map_location=torch.device(args.device), weights_only=False)

    ckpt_args = checkpoint.get('args', {})
    if ckpt_args:
        print(f"  从检查点读取训练参数:")
    else:
        print(f"  [警告] 检查点中未找到训练参数，使用默认值")

    def get_param(name, default_value, cmd_value):
        if cmd_value is not None:
            return cmd_value
        if name in ckpt_args:
            return ckpt_args[name]
        return default_value

    data_name = get_param('data_name', 'unsw', args.data_name)
    num_layers = get_param('num_layers', 3, args.num_layers)
    knn_k = get_param('knn_k', 25, args.knn_k)
    topk_pool = get_param('topk_pool', 50, args.topk_pool)
    tensor_neurons = get_param('tensor_neurons', 32, args.tensor_neurons)
    dropout_rate = get_param('dropout_rate', 0.3, args.dropout_rate)
    gnn_dropout = get_param('gnn_dropout', 0.1, args.gnn_dropout)
    drop_edge_prob = get_param('drop_edge_prob', 0.1, args.drop_edge_prob)
    feature_jitter = get_param('feature_jitter', 0.05, args.feature_jitter)
    fps_alpha = get_param('fps_alpha', 0.2, args.fps_alpha)
    attn_dropout = get_param('attn_dropout', 0.5, args.attn_dropout)
    ntn_temperature = get_param('ntn_temperature', 1.0, args.ntn_temperature)
    l2_temperature = get_param('l2_temperature', 1.0, args.l2_temperature)
    model_use_ntn = get_param('use_ntn', True, None)
    eval_use_ntn = model_use_ntn if args.eval_use_ntn is None else args.eval_use_ntn
    disable_topology_score = get_param('disable_topology_score', False, args.disable_topology_score)
    graph_backbone = get_param('graph_backbone', 'hgnn', args.graph_backbone)

    print(f"  数据集: {data_name}")
    print(f"  模型配置: num_layers={num_layers}, topk_pool={topk_pool}, use_ntn(model)={model_use_ntn}")
    print(f"  测试配置: use_ntn(test)={eval_use_ntn}, report_both_routes={args.report_both_routes}")
    print(f"  图骨干: {graph_backbone}")

    print(f"[2/4] 创建模型...")
    model = FingerprintNet(
        feat_dim=128,
        num_layers=num_layers,
        knn_k=knn_k,
        topk_pool=topk_pool,
        attn_dropout=attn_dropout,
        fps_alpha=fps_alpha,
        tensor_neurons=tensor_neurons,
        dropout_rate=dropout_rate,
        gnn_dropout=gnn_dropout,
        drop_edge_prob=drop_edge_prob,
        feature_jitter=feature_jitter,
        ntn_temperature=ntn_temperature,
        l2_temperature=l2_temperature,
        use_ntn=model_use_ntn,
        pretrained_backbone=False,
        disable_topology_score=disable_topology_score,
        graph_backbone=graph_backbone,
    )

    _ = load_checkpoint_into_model(
        model,
        args.ckpt_path,
        device=torch.device(args.device),
    )

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # 运行推理时间对比测试
    benchmark_inference_time(model, device)

    print(f"[3/4] 准备数据...")
    all_files, cfpose_files, unsw_files = collect_all_files(data_name, DATA_CONFIG)

    if data_name == 'unsw':
        train_files, val_files, test_files, _ = split_files_fixed_uwa(all_files)
    elif data_name == 'cfpose':
        train_files, val_files, test_files, _ = split_files_cfpose(
            all_files,
            test_count=args.cfpose_test_count,
            val_count=args.cfpose_val_count,
            train_count=args.cfpose_train_count,
        )
    elif data_name == 'mixed':
        train_files, val_files, test_files, _ = split_files_mixed_protocol(
            cfpose_files,
            unsw_files,
            cfpose_test_count=args.cfpose_test_count,
            cfpose_val_count=args.cfpose_val_count,
            cfpose_train_count=args.cfpose_train_count,
        )
    else:
        print(f"[错误] 不支持的数据集: {data_name}")
        sys.exit(1)

    print(f"  测试集样本数: {len(test_files)}")

    benchmark_ds = FingerprintDataset(
        None,
        patch_size=32,
        stride=16,
        mode='test',
        files=test_files,
        data_name=data_name
    )
    benchmark_loader = data_cls.DataLoader(
        benchmark_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print(f"[4/4] 运行 Benchmark 测试...")
    print("")

    if data_name == 'mixed':
        print("="*70)
        print("UNSW 数据集测试")
        print("="*70)
        unsw_eer, unsw_thr, unsw_tp, unsw_tn, unsw_fp, unsw_fn, unsw_tar_1e4, unsw_rank1, unsw_rank5, unsw_rank10 = test_benchmark(
            model,
            benchmark_loader,
            device,
            dataset_name='unsw',
            ckpt_dir=os.path.dirname(args.ckpt_path),
            use_ntn=eval_use_ntn,
            report_both_routes=args.report_both_routes,
            enable_visualization=args.enable_visualization,
            enable_score_export=args.enable_score_export,
            enable_det_curve=args.enable_det_curve,
            enable_score_histogram=args.enable_score_histogram,
        )

        print("="*70)
        print("CFPose 数据集测试")
        print("="*70)
        cfpose_eer, cfpose_thr, cfpose_tp, cfpose_tn, cfpose_fp, cfpose_fn, cfpose_tar_1e4, cfpose_rank1, cfpose_rank5, cfpose_rank10 = test_benchmark(
            model,
            benchmark_loader,
            device,
            dataset_name='cfpose',
            ckpt_dir=os.path.dirname(args.ckpt_path),
            use_ntn=eval_use_ntn,
            report_both_routes=args.report_both_routes,
            enable_visualization=args.enable_visualization,
            enable_score_export=args.enable_score_export,
            enable_det_curve=args.enable_det_curve,
            enable_score_histogram=args.enable_score_histogram,
            gallery_strategy=args.gallery_strategy,
            gallery_metadata=args.gallery_metadata,
        )

        n_unsw = len([f for f in test_files if 'unsw' in f.lower()])
        n_cfpose = len([f for f in test_files if 'cfpose' in f.lower()])
        total = n_unsw + n_cfpose
        weighted_eer = (unsw_eer * n_unsw + cfpose_eer * n_cfpose) / total if total > 0 else float('nan')

        print("\n" + "="*70)
        print("混合数据集测试结果汇总")
        print("="*70)
        print(f"UNSW EER:    {unsw_eer*100:.4f}% (TAR@FAR=0.01%: {unsw_tar_1e4:.6f})")
        if np.isfinite(unsw_rank1):
            print(f"UNSW Rank-1: {unsw_rank1:.6f} ({unsw_rank1*100:.2f}%)")
        print(f"CFPose EER:  {cfpose_eer*100:.4f}% (TAR@FAR=0.01%: {cfpose_tar_1e4:.6f})")
        if np.isfinite(cfpose_rank1):
            print(f"CFPose Rank-1: {cfpose_rank1:.6f} ({cfpose_rank1*100:.2f}%)")
        print(f"加权平均 EER: {weighted_eer*100:.4f}%")
        print("="*70)
    else:
        eer, eer_threshold, TP, TN, FP, FN, tar_1e4, rank1, rank5, rank10 = test_benchmark(
            model,
            benchmark_loader,
            device,
            dataset_name=data_name,
            ckpt_dir=os.path.dirname(args.ckpt_path),
            use_ntn=eval_use_ntn,
            report_both_routes=args.report_both_routes,
            enable_visualization=args.enable_visualization,
            enable_score_export=args.enable_score_export,
            enable_det_curve=args.enable_det_curve,
            enable_score_histogram=args.enable_score_histogram,
            gallery_strategy=args.gallery_strategy,
            gallery_metadata=args.gallery_metadata,
        )

        print("\n" + "="*70)
        print("测试结果汇总")
        print("="*70)
        print(f"EER:           {eer*100:.4f}%")
        print(f"TAR@FAR=0.01%: {tar_1e4:.6f}")
        if data_name in ('unsw', 'cfpose') and np.isfinite(rank1):
            print(f"Rank-1:        {rank1:.6f} ({rank1*100:.2f}%)")
            print(f"Rank-5:        {rank5:.6f} ({rank5*100:.2f}%)")
            print(f"Rank-10:       {rank10:.6f} ({rank10*100:.2f}%)")
        print("="*70)

    print("\n测试完成!")


if __name__ == '__main__':
    main()
