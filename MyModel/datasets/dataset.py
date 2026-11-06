import os
import sys

import glob
import random
import math
import numpy as np
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import torch.utils.data as data_cls
from MyModel.utils.utils import get_finger_id, sort_files_for_dataset


class RandomIlluminationGradient:
    def __init__(self, prob=0.5, strength_range=(0.2, 0.8), gamma_range=(0.8, 1.2)):
        self.prob = min(max(float(prob), 0.0), 1.0)
        s_min = float(strength_range[0])
        s_max = float(strength_range[1])
        self.strength_range = (min(s_min, s_max), max(s_min, s_max))
        g_min = max(float(gamma_range[0]), 1e-3)
        g_max = max(float(gamma_range[1]), 1e-3)
        self.gamma_range = (min(g_min, g_max), max(g_min, g_max))

    def __call__(self, img_tensor):
        """
        img_tensor: [C, H, W] 且值域在 [0, 1]。
        """
        if random.random() > self.prob:
            return img_tensor

        if not torch.is_tensor(img_tensor) or img_tensor.ndim != 3:
            return img_tensor

        _, H, W = img_tensor.shape
        if H <= 1 or W <= 1:
            return img_tensor

        y = torch.linspace(-1.0, 1.0, H, dtype=img_tensor.dtype, device=img_tensor.device).view(H, 1).expand(H, W)
        x = torch.linspace(-1.0, 1.0, W, dtype=img_tensor.dtype, device=img_tensor.device).view(1, W).expand(H, W)

        angle = random.uniform(0.0, 2.0 * math.pi)
        gradient = x * math.cos(angle) + y * math.sin(angle)
        grad_min = gradient.min()
        grad_max = gradient.max()
        denom = grad_max - grad_min
        if torch.abs(denom) < 1e-6:
            return img_tensor
        gradient = (gradient - grad_min) / denom

        strength = random.uniform(self.strength_range[0], self.strength_range[1])
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        illumination_map = torch.pow(gradient, gamma)

        light_mask = (1.0 - strength) + strength * illumination_map
        out_tensor = img_tensor * light_mask.unsqueeze(0)
        return torch.clamp(out_tensor, 0.0, 1.0)


class FingerprintDataset(data_cls.Dataset):
    def __init__(
        self,
        root_dir=None,
        patch_size=32,
        stride=16,
        mode='train',
        files=None,
        augment=False,
        data_name='cfpose',
        use_minu_metadata=True,
        knn_k=25,
        augment_profile='default',
        micro_rotate_deg=5.0,
        micro_translate_ratio=0.02,
        enable_random_illum_gradient=False,
        illum_gradient_prob=0.5,
        illum_gradient_strength_min=0.08,
        illum_gradient_strength_max=0.30,
        illum_gradient_gamma_min=0.7,
        illum_gradient_gamma_max=1.8,
    ):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.data_name = data_name
        self.use_minu_metadata = bool(use_minu_metadata)
        self.knn_k = knn_k
        self.augment_profile = str(augment_profile).lower().strip()
        if self.augment_profile not in ('default', 'micro'):
            raise ValueError(f'Unsupported augment_profile: {augment_profile}')
        self.micro_rotate_deg = max(float(micro_rotate_deg), 0.0)
        self.micro_translate_ratio = max(float(micro_translate_ratio), 0.0)
        self.enable_random_illum_gradient = bool(enable_random_illum_gradient)
        self.illum_gradient_prob = min(max(float(illum_gradient_prob), 0.0), 1.0)
        self.illum_gradient_strength_min = float(illum_gradient_strength_min)
        self.illum_gradient_strength_max = float(illum_gradient_strength_max)
        if self.illum_gradient_strength_min > self.illum_gradient_strength_max:
            self.illum_gradient_strength_min, self.illum_gradient_strength_max = (
                self.illum_gradient_strength_max,
                self.illum_gradient_strength_min,
            )
        self.illum_gradient_gamma_min = max(float(illum_gradient_gamma_min), 1e-3)
        self.illum_gradient_gamma_max = max(float(illum_gradient_gamma_max), 1e-3)
        if self.illum_gradient_gamma_min > self.illum_gradient_gamma_max:
            self.illum_gradient_gamma_min, self.illum_gradient_gamma_max = (
                self.illum_gradient_gamma_max,
                self.illum_gradient_gamma_min,
            )
        
        # 自动检测文件类型
        if files is None:
            self.files = glob.glob(os.path.join(root_dir, '*.mat'))
            sort_files_for_dataset(self.files, self.data_name)
        else:
            self.files = list(files)
        
        # 光度增强链路统一在 Tensor 空间执行，便于接入方向性光照梯度。
        illum_prob = self.illum_gradient_prob if self.enable_random_illum_gradient else 0.0
        self.photometric_aug = transforms.Compose([
            transforms.RandomApply([
                RandomIlluminationGradient(
                    prob=illum_prob,
                    strength_range=(self.illum_gradient_strength_min, self.illum_gradient_strength_max),
                    gamma_range=(self.illum_gradient_gamma_min, self.illum_gradient_gamma_max),
                )
            ], p=1.0),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 7), sigma=(0.1, 1.0))], p=0.3),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
        ])
        
        # 遮挡增强：模拟反光盲区或残缺
        self.erasing_aug = transforms.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        
        # 对整张图进行变换
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def _get_data_name_for_file(self, file_path):
        """根据文件路径判断数据集名称，支持混合数据集"""
        if self.data_name != 'mixed':
            return self.data_name
        # 混合模式下，根据路径判断
        if 'cfpose' in file_path.lower():
            return 'cfpose'
        elif 'unsw' in file_path.lower():
            return 'unsw'
        else:
            # 默认返回 cfpose
            return 'cfpose'

    def __len__(self):
        return len(self.files)

    def _load_mat(self, file_path):
        data = sio.loadmat(file_path)
        if 'img' not in data or 'mask' not in data:
            return None, None
        img = data['img'].astype(np.float32)
        mask = data['mask'].astype(np.float32)

        if img.max() <= 1.0:
            img = img * 255.0
        return img, mask

    def _get_patch_coords(self, img, mask):
        H, W = img.shape
        ps = self.patch_size
        st = self.stride
        
        # 1. 生成网格坐标（向量化）
        y_coords = np.arange(0, H - ps + 1, st)
        x_coords = np.arange(0, W - ps + 1, st)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算中心坐标并归一化
        center_x = xx + ps // 2
        center_y = yy + ps // 2
        norm_x = ((center_x + 0.5) / W) * 2 - 1
        norm_y = ((center_y + 0.5) / H) * 2 - 1
        
        coords = np.stack([norm_x.ravel(), norm_y.ravel()], axis=1)
        
        # 2. 使用积分图加速计算 mask 均值
        # 添加 padding 避免边界条件检查
        padded_mask = np.pad(mask, ((1, 0), (1, 0)), mode='constant')
        # 积分图：sum[y,x] = sum_{i<y,j<x} mask[i,j]
        integral_mask = np.cumsum(np.cumsum(padded_mask, axis=0), axis=1)
        
        # 计算每个 patch 的 sum
        # patch 区域：[y:y+ps, x:x+ps]
        # 使用积分图公式：sum = I[y2+1,x2+1] - I[y1,x2+1] - I[y2+1,x1] + I[y1,x1]
        y1 = yy
        x1 = xx
        y2 = yy + ps - 1
        x2 = xx + ps - 1
        
        sum_patch = (integral_mask[y2 + 1, x2 + 1] - 
                     integral_mask[y1, x2 + 1] - 
                     integral_mask[y2 + 1, x1] + 
                     integral_mask[y1, x1])
        
        mean_patch = sum_patch / (ps * ps)
        node_mask = (mean_patch >= 0.5).ravel()
        
        if len(coords) == 0:
            return None, None
            
        return (
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(node_mask, dtype=torch.bool)
        )

    def _apply_augmentation(self, img, mask):
        """同时对图像、Mask 和 细节点坐标进行复合增强。"""
        H, W = img.shape
        # 转换成 PIL 图像方便调用 torchvision.functional
        img_pil = Image.fromarray(img.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        if self.augment_profile == 'micro':
            angle = random.uniform(-self.micro_rotate_deg, self.micro_rotate_deg)
            max_dx = W * self.micro_translate_ratio
            max_dy = H * self.micro_translate_ratio
            tx = random.uniform(-max_dx, max_dx)
            ty = random.uniform(-max_dy, max_dy)
            scale = 1.0
            shear = 0.0

            img_aug = TF.affine(
                img_pil,
                angle,
                (tx, ty),
                scale,
                shear,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            mask_aug = TF.affine(
                mask_pil,
                angle,
                (tx, ty),
                scale,
                shear,
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            img_tensor = TF.to_tensor(img_aug)
            img_tensor = self.photometric_aug(img_tensor)
            img_np = torch.clamp(img_tensor, 0.0, 1.0).squeeze(0).numpy() * 255.0
            img_np = img_np.astype(np.float32)
            return img_np, np.array(mask_aug).astype(np.float32)

        # 1. 几何变换参数 (旋转、平移、缩放)
        angle = random.uniform(-45, 45)
        max_dx = W * 0.1
        max_dy = H * 0.1
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale = random.uniform(0.8, 1.2)   # 模拟手指相对于摄像头的远近
        shear = 0.0    # 无剪切

        # 变换图像和 Mask (插值方式不同)
        img_aug = TF.affine(img_pil, angle, (tx, ty), scale, shear, 
                            interpolation=transforms.InterpolationMode.BILINEAR)
        mask_aug = TF.affine(mask_pil, angle, (tx, ty), scale, shear,
                             interpolation=transforms.InterpolationMode.NEAREST)

        # 2. 随机光度增强（包含方向性光照梯度）
        img_tensor = TF.to_tensor(img_aug)
        img_tensor = self.photometric_aug(img_tensor)
        img_np = torch.clamp(img_tensor, 0.0, 1.0).squeeze(0).numpy() * 255.0
        img_np = img_np.astype(np.float32)
        
        # 3. 随机高斯噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(2, 8), img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 255)

        return img_np, np.array(mask_aug).astype(np.float32)

    def __getitem__(self, index):
        file_path = self.files[index]
        
        img, mask = self._load_mat(file_path)
        
        if img is None:
            return self.__getitem__((index + 1) % len(self.files))

        # 执行数据增强
        if self.augment:
            img, mask = self._apply_augmentation(img, mask)

        # 获取中心坐标、标签和 node_mask
        coords, node_mask = self._get_patch_coords(img, mask)
        
        if coords is None:
            return self.__getitem__((index + 1) % len(self.files))

        # 预计算 KNN 邻接矩阵 W
        N = coords.size(0)
        dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [N, N]
        
        # 让被 mask 的节点在 KNN 阶段就全部断联
        # 1. 无效节点的行和列都设为无穷大，这样它们不会被选为邻居
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(0)  # 只有有效节点之间才可能有连接
        dist = torch.where(node_mask_expanded, dist, torch.tensor(1e9, device=dist.device))
        
        k = min(self.knn_k, N)
        if k > 0:
            val, idx = torch.topk(dist, k=k, dim=-1, largest=False)
            stride_len = val[:, 1].median()
            radius_thresh = 3.0 * stride_len
            mask_knn = val < radius_thresh
            W = torch.zeros_like(dist)
            W.scatter_(1, idx, mask_knn.float())
            
            # 额外确保：最终 W 中只有有效节点之间才有连接
            W = W * node_mask_expanded.float()
        else:
            W = torch.zeros(N, N)

        # 将整张图转为 Tensor
        full_img = self.img_transform(img.astype(np.uint8))
        
        # 5. 随机擦除 (模拟遮挡/残缺)
        if self.augment and self.augment_profile == 'default':
            full_img = self.erasing_aug(full_img)

        file_data_name = self._get_data_name_for_file(file_path)
        return {
            'full_img': full_img,       # [1, H, W]
            'coords': coords,           # [N, 2] 归一化坐标 ([-1, 1])
            'node_mask': node_mask,     # [N] 布尔掩码，标记有效节点
            'W': W,                     # [N, N] 预计算的邻接矩阵
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'id': get_finger_id(file_path, file_data_name)
        }


class BalancedBatchSampler(data_cls.Sampler):
    def __init__(self, labels, n_classes, n_samples, num_batches, seed=0, domain_labels=None, unsw_sampling_ratio=0.7):
        self.labels = list(labels)
        self.n_classes = int(n_classes)
        self.n_samples = int(n_samples)
        self.num_batches = int(num_batches)
        self.base_seed = int(seed)
        self.current_epoch = 0
        self.domain_labels = list(domain_labels) if domain_labels is not None else None
        self.unsw_sampling_ratio = float(unsw_sampling_ratio)

        id_to_indices = {}
        id_to_domain = {}
        for idx, lab in enumerate(self.labels):
            id_to_indices.setdefault(int(lab), []).append(idx)
            if self.domain_labels is not None:
                domain = self.domain_labels[idx]
                if int(lab) not in id_to_domain:
                    id_to_domain[int(lab)] = domain
                elif id_to_domain[int(lab)] != domain:
                    raise RuntimeError(f'Inconsistent domain labels found for id={lab}.')

        # keep ids with at least 2 samples so positives exist
        self.id_to_indices = {k: v for k, v in id_to_indices.items() if len(v) >= 2}
        self.valid_ids = sorted(self.id_to_indices.keys())
        if len(self.valid_ids) < 2:
            raise RuntimeError('Need >=2 identities with >=2 samples each for balanced batches.')

        self.id_to_domain = {k: id_to_domain[k] for k in self.valid_ids if k in id_to_domain}
        self.unsw_ids = sorted([sid for sid in self.valid_ids if self.id_to_domain.get(sid) == 'unsw'])
        self.cfpose_ids = sorted([sid for sid in self.valid_ids if self.id_to_domain.get(sid) == 'cfpose'])

        if self.n_classes > len(self.valid_ids):
            raise RuntimeError(f'n_classes={self.n_classes} is larger than available ids={len(self.valid_ids)}')

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.base_seed + self.current_epoch)
        for _ in range(self.num_batches):
            batch_indices = []

            if self.domain_labels is None or len(self.unsw_ids) == 0 or len(self.cfpose_ids) == 0:
                chosen_ids = rng.sample(self.valid_ids, self.n_classes)
            else:
                n_unsw = int(round(self.n_classes * self.unsw_sampling_ratio))
                n_unsw = max(1, min(n_unsw, self.n_classes - 1))
                n_cfpose = self.n_classes - n_unsw

                # Fallback protection when one domain has too few valid IDs.
                if len(self.unsw_ids) < n_unsw:
                    n_unsw = len(self.unsw_ids)
                    n_cfpose = self.n_classes - n_unsw
                if len(self.cfpose_ids) < n_cfpose:
                    n_cfpose = len(self.cfpose_ids)
                    n_unsw = self.n_classes - n_cfpose

                if n_unsw <= 0 or n_cfpose <= 0:
                    chosen_ids = rng.sample(self.valid_ids, self.n_classes)
                else:
                    chosen_unsw = rng.sample(self.unsw_ids, n_unsw)
                    chosen_cfpose = rng.sample(self.cfpose_ids, n_cfpose)
                    chosen_ids = chosen_unsw + chosen_cfpose

            for sample_id in chosen_ids:
                candidates = self.id_to_indices[sample_id]
                if len(candidates) >= self.n_samples:
                    picked = rng.sample(candidates, self.n_samples)
                else:
                    picked = [rng.choice(candidates) for _ in range(self.n_samples)]
                batch_indices.extend(picked)
            rng.shuffle(batch_indices)
            yield batch_indices