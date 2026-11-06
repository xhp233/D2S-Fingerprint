import os
import json
import shutil
import numpy as np
import torch

def _atomic_write_json(obj, path: str):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class CheckpointManager:
    """Keep a small set of checkpoints (last + top-K best) to save disk."""

    def __init__(self, ckpt_dir: str, keep_last: bool = True, save_best_k: int = 3, mode: str = 'min'):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.keep_last = bool(keep_last)
        self.save_best_k = int(save_best_k)
        self.mode = mode  # 'min' or 'max'
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.ckpt_dir, 'manifest.json')

        # list of dicts: {path, metric, epoch}
        self.best = []
        self.milestones = {} # stage_name -> path
        self.last_path = os.path.join(self.ckpt_dir, 'last.pt')
        self.best_link = os.path.join(self.ckpt_dir, 'best.pt')
        # 窗口最佳检查点：窗口ID -> {path, metric, epoch}
        self.window_best = {}

        self._load_manifest_if_any()

    def _load_manifest_if_any(self):
        if not os.path.exists(self.manifest_path):
            return
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            self.best = m.get('best', [])
            self.milestones = m.get('milestones', {})
            self.window_best = m.get('window_best', {})
            # 将相对路径转换为绝对路径
            for rec in self.best:
                if not os.path.isabs(rec['path']):
                    rec['path'] = os.path.abspath(rec['path'])
            for k, v in self.milestones.items():
                if not os.path.isabs(v):
                    self.milestones[k] = os.path.abspath(v)
            for k, rec in self.window_best.items():
                if not os.path.isabs(rec['path']):
                    rec['path'] = os.path.abspath(rec['path'])
        except Exception:
            # if manifest is corrupt, just start fresh
            self.best = []
            self.milestones = {}
            self.window_best = {}

    def _sort_best(self):
        reverse = (self.mode == 'max')
        self.best.sort(key=lambda x: x['metric'], reverse=reverse)

    def _better_than(self, a: float, b: float) -> bool:
        return (a > b) if self.mode == 'max' else (a < b)

    def _save_blob(self, payload: dict, path: str):
        tmp = path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _update_best_symlink(self):
        if len(self.best) == 0:
            return
        best_path = self.best[0]['path']
        # best.pt as symlink if possible, else copy
        try:
            if os.path.islink(self.best_link) or os.path.exists(self.best_link):
                os.remove(self.best_link)
            os.symlink(os.path.basename(best_path), self.best_link)
        except Exception:
            try:
                shutil.copy2(best_path, self.best_link)
            except Exception:
                pass

    def _prune_best(self):
        self._sort_best()
        if self.save_best_k <= 0:
            # delete all best checkpoints
            for rec in self.best:
                try:
                    if os.path.exists(rec['path']):
                        os.remove(rec['path'])
                except Exception:
                    pass
            self.best = []
            return

        while len(self.best) > self.save_best_k:
            rec = self.best.pop(-1)
            try:
                if os.path.exists(rec['path']):
                    os.remove(rec['path'])
            except Exception:
                pass

    def save_last(self, payload: dict):
        if not self.keep_last:
            return
        self._save_blob(payload, self.last_path)

    def save_milestones(self, payload: dict, stage_names: list, epoch: int):
        """Save a physical milestone for an epoch and create symbolic links for alias names."""
        if not stage_names:
            return
            
        # 1. 物理文件名以 Epoch 命名，确保唯一性且不因阶段名重复而多次保存
        phys_name = f'epoch_{epoch:04d}_milestone.pt'
        phys_path = os.path.join(self.ckpt_dir, phys_name)
        
        # 物理保存一次
        if not os.path.exists(phys_path):
            self._save_blob(payload, phys_path)
            
        # 2. 为每个阶段名创建软链接（或复制），这样既有语义化名称又不占额外空间
        for name in stage_names:
            alias_name = f'milestone_{name}.pt'
            alias_path = os.path.join(self.ckpt_dir, alias_name)
            
            try:
                if os.path.islink(alias_path) or os.path.exists(alias_path):
                    os.remove(alias_path)
                # 在 Linux 系统下优先使用软链接节省空间
                os.symlink(phys_name, alias_path)
            except Exception:
                # 兼容性备选：如果系统不支持软链接，则执行拷贝
                shutil.copy2(phys_path, alias_path)
                
            self.milestones[name] = alias_path
            print(f'[CKPT] Milestone alias created: {alias_name} -> {phys_name}')

    def maybe_save_best(self, payload: dict, metric: float, epoch: int, prefix: str = 'epoch', mixed_eers: dict = None, tar_1e4: float = None):
        if not np.isfinite(metric):
            return False
        if self.save_best_k <= 0:
            return False

        self._sort_best()
        qualifies = False
        if len(self.best) < self.save_best_k:
            qualifies = True
        else:
            worst = self.best[-1]['metric']
            qualifies = self._better_than(metric, worst)

        if not qualifies:
            return False

        # 构建文件名
        if mixed_eers and 'unsw' in mixed_eers and 'cfpose' in mixed_eers:
            # 混合数据集情况：包含两个数据集的 EER
            unsw_eer = mixed_eers['unsw']
            cfpose_eer = mixed_eers['cfpose']
            if tar_1e4 is not None:
                fname = f'{prefix}{epoch:04d}_metric{metric:.6f}_tar{tar_1e4:.6f}_unsw{unsw_eer:.6f}_cfpose{cfpose_eer:.6f}.pt'
            else:
                fname = f'{prefix}{epoch:04d}_metric{metric:.6f}_unsw{unsw_eer:.6f}_cfpose{cfpose_eer:.6f}.pt'
        else:
            # 单数据集情况：包含EER和TAR@FAR=0.01%
            if tar_1e4 is not None:
                fname = f'{prefix}{epoch:04d}_metric{metric:.6f}_tar{tar_1e4:.6f}.pt'
            else:
                fname = f'{prefix}{epoch:04d}_metric{metric:.6f}.pt'
            
        path = os.path.join(self.ckpt_dir, fname)
        self._save_blob(payload, path)
        self.best.append({'path': path, 'metric': float(metric), 'epoch': int(epoch)})
        self._prune_best()
        self._sort_best()
        self._update_best_symlink()
        return True

    def maybe_save_window_best(self, window_id: str, payload: dict, metric: float, epoch: int, mixed_eers: dict = None):
        """保存指定窗口的最佳检查点"""
        if not np.isfinite(metric):
            return False

        # 检查是否比当前窗口最佳更好
        current_window_best = self.window_best.get(window_id)
        if current_window_best is None:
            # 窗口还没有最佳，直接保存
            qualifies = True
        else:
            qualifies = self._better_than(metric, current_window_best['metric'])

        if not qualifies:
            return False

        # 构建文件名
        if mixed_eers and 'unsw' in mixed_eers and 'cfpose' in mixed_eers:
            # 混合数据集情况：包含两个数据集的 EER
            unsw_eer = mixed_eers['unsw']
            cfpose_eer = mixed_eers['cfpose']
            fname = f'window_{window_id}_epoch{epoch:04d}_metric{metric:.6f}_unsw{unsw_eer:.6f}_cfpose{cfpose_eer:.6f}.pt'
        else:
            # 单数据集情况：保持原有格式
            fname = f'window_{window_id}_epoch{epoch:04d}_metric{metric:.6f}.pt'
            
        path = os.path.join(self.ckpt_dir, fname)
        self._save_blob(payload, path)

        # 更新窗口最佳记录
        if current_window_best is not None:
            # 删除旧的窗口最佳文件
            try:
                if os.path.exists(current_window_best['path']):
                    os.remove(current_window_best['path'])
            except Exception:
                pass

        self.window_best[window_id] = {'path': path, 'metric': float(metric), 'epoch': int(epoch)}
        return True

    def write_manifest(self, extra: dict = None):
        m = {
            'ckpt_dir': self.ckpt_dir,
            'mode': self.mode,
            'keep_last': self.keep_last,
            'save_best_k': self.save_best_k,
            'best': self.best,
            'milestones': self.milestones,
            'window_best': self.window_best,
            'last': self.last_path if self.keep_last else '',
        }
        if extra:
            m.update(extra)
        _atomic_write_json(m, self.manifest_path)

