import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from MyModel.utils.utils import compute_gf_score_from_H

from .HyperGCTcore import HGNN


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=32.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features    # 128
        self.out_features = out_features  # 训练集总 ID 数
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. 归一化线性层：得到 cos(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # 2. 算 sin(theta)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # 3. cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        # 4. 边界处理
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # 5. 构造 one-hot 并应用 margin
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SimpleGCN(nn.Module):
    """A lightweight GCN backbone for ablation (pairwise graph only)."""
    def __init__(self, in_channel=128, n_emb_dims=128, num_layers=3):
        super().__init__()
        self.num_layers = int(max(1, num_layers))
        layers = []
        for i in range(self.num_layers):
            in_dim = in_channel if i == 0 else n_emb_dims
            layers.append(nn.Linear(in_dim, n_emb_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, h, W):
        # h: [B, C, N], W: [B, N, N]
        x = h.transpose(1, 2)  # [B, N, C]
        B, N, _ = x.shape

        eye = torch.eye(N, dtype=W.dtype, device=W.device).unsqueeze(0)
        A = torch.clamp(W, min=0.0) + eye
        D_inv = 1.0 / (A.sum(dim=-1, keepdim=True) + 1e-12)
        A_norm = D_inv * A

        for i, layer in enumerate(self.layers):
            x = torch.bmm(A_norm, x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        feat_final = x.transpose(1, 2)  # [B, C, N]
        return feat_final, A


class NodeEncoder(nn.Module):
    def __init__(self, feat_dim=128, dropout_rate=0.3, pretrained=True):
        super().__init__()
        # CNN Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 输入为整张图。
        # 1. 修改输入通道并移除初期的 MaxPool (将下采样率由 4 降为 2)，保留更多低层纹理。
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.maxpool = nn.Identity() 
        
        # 2. 弃用下采样过大的 Layer4 (Stride 32)，改用 Layer3 输出 (Stride 8)。
        # 指纹特征更依赖中层的几何拓扑，不需要 Layer4 的超高层语义抽象。
        # 此时总 Stride = 2(conv1) * 1(layer1) * 2(layer2) * 2(layer3) = 8。
        # 原图 480x640 -> 特征图 60x80 (Stride 8)。
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        
        # 3. 将 Layer3 的输出通道 (256) 降低到指定的 feat_dim
        self.cnn_proj = nn.Conv2d(256, feat_dim, kernel_size=1)
        
        # 执行分层冻结
        self._freeze_stages()

    def enable_conv1_finetuning(self):
        """专用于后期突破上限的微调接口，仅解冻随机初始化的 Conv1"""
        print("[NodeEncoder] WARNING: Unfreezing randomized Conv1 for fine-tuning!")
        # 优先兼容历史实现：conv1 挂在 self.layer0[0]
        if hasattr(self, 'layer0'):
            try:
                for param in self.layer0[0].parameters():
                    param.requires_grad = True
                return
            except Exception:
                pass

        # 当前实现中 conv1 位于 backbone[0]
        if hasattr(self, 'backbone') and isinstance(self.backbone, nn.Sequential) and len(self.backbone) > 0:
            for param in self.backbone[0].parameters():
                param.requires_grad = True

    def _freeze_stages(self):
        # 1. 冻结所有 backbone 参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. 遍历网络，仅解冻所有的 BatchNorm2d 层
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, full_img, coords):
        feat_map = self.backbone(full_img) # [B, 256, h, w]
        feat_map = self.cnn_proj(feat_map) # [B, feat_dim, h, w]
        
        # 使用 grid_sample 提取特征。grid shape: [B, N, 1, 2]
        grid = coords.unsqueeze(2) 
        v_feat = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=False) # [B, feat_dim, N, 1]
        v_feat = v_feat.squeeze(-1).transpose(1, 2) # [B, N, feat_dim]
        
        # 展平 batch 和 node 维度
        B, N, C = v_feat.shape
        v_feat_flat = v_feat.reshape(B * N, C)
        
        return v_feat_flat # [B*N, feat_dim]


class InteractionScorer(nn.Module):
    def __init__(
        self,
        feat_dim,
        tensor_neurons=64,
        dropout_rate=0.3,
        ntn_temperature=1.0,
        l2_temperature=1.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.tensor_neurons = tensor_neurons
        self.ntn_temperature = max(float(ntn_temperature), 1e-4)
        self.l2_temperature = max(float(l2_temperature), 1e-4)
        
        # 1. 双线性张量 W [feat_dim, feat_dim, tensor_neurons]
        self.weight_matrix = nn.Parameter(torch.Tensor(feat_dim, feat_dim, tensor_neurons))
        
        # 2. 拼接向量的线性变换 W_block [tensor_neurons, 2*feat_dim]
        self.weight_matrix_block = nn.Parameter(torch.Tensor(tensor_neurons, 2 * feat_dim))
        
        # 3. 偏置项
        self.bias = nn.Parameter(torch.Tensor(tensor_neurons, 1))
        
        # 4. Scoring Layer
        self.scoring_layer = nn.Sequential(
            nn.Linear(tensor_neurons, tensor_neurons),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(tensor_neurons, 1)
        )
        
        # 5. 可学习的融合权重 Alpha (控制 Cosine 和 NTN 的比例)
        self.alpha = nn.Parameter(torch.tensor([0.0]))
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.xavier_uniform_(self.weight_matrix_block)
        nn.init.xavier_uniform_(self.bias)
    
    def forward(self, z1, z2):
        # 处理 1D 输入（单个样本）
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)  # [1, feat_dim]
            z2 = z2.unsqueeze(0)  # [1, feat_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = z1.size(0)
        
        # 1. 双线性项：scoring = (z1^T * W) * z2
        scoring = torch.matmul(z1, self.weight_matrix.view(self.feat_dim, -1))
        scoring = scoring.view(batch_size, self.feat_dim, -1).permute([0, 2, 1])
        scoring = torch.matmul(
            scoring, z2.view(batch_size, self.feat_dim, 1)
        ).view(batch_size, -1)  # shape: [B, tensor_neurons]
        
        # 2. 拼接项：block_scoring = W_block * [z1, z2]
        combined_representation = torch.cat((z1, z2), dim=1)  # [B, 2*feat_dim]
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )  # shape: [B, tensor_neurons]
        
        # 3. NTN 输出：scores = ReLU(scoring + block_scoring + bias)
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        
        # 4. Scoring Layer 得到最终分数
        ntn_score = self.scoring_layer(scores).squeeze(-1)  # shape: [B]
        
        # 计算 L2 距离
        l2_score = torch.norm(z1 - z2, dim=-1)
        
        # 动态融合
        # sigmoid(alpha) 保证权重在 0~1 之间
        weight = torch.sigmoid(self.alpha)
        
        # 温度缩放：平滑高分区，避免分数过早饱和到 1.0 附近。
        ntn_sim = torch.sigmoid(ntn_score / self.ntn_temperature)
        l2_sim = torch.exp(-(l2_score / self.l2_temperature))
        
        # 融合：alpha * NTN + (1-alpha) * L2
        final_score = weight * ntn_sim + (1.0 - weight) * l2_sim
        
        # 如果输入是 1D，输出也 squeeze 成标量
        if squeeze_output:
            final_score = final_score.squeeze(0)  # shape: []
        
        return final_score


class FingerprintNet(nn.Module):
    def __init__(
        self,
        feat_dim=128,
        num_layers=3,
        knn_k=25,
        topk_pool=30,
        attn_dropout=0.5,
        fps_alpha=0.2,
        tensor_neurons=64,
        dropout_rate=0.3,
        gnn_dropout=0.1,
        drop_edge_prob=0.1,
        feature_jitter=0.05,
        ntn_temperature=1.0,
        l2_temperature=1.0,
        use_ntn=True,
        disable_topology_score=False,
        graph_backbone='hgnn',
        disable_contraction=False,
        pretrained_backbone=True,
    ):
        super().__init__()
        self.encoder = NodeEncoder(feat_dim, dropout_rate=dropout_rate, pretrained=pretrained_backbone)
        self.graph_backbone = str(graph_backbone).lower()
        self.disable_topology_score = bool(disable_topology_score)
        if self.graph_backbone == 'hgnn':
            # 直接复用 HyperGCT 的超图骨干逻辑 (HGNN)
            self.gnn = HGNN(in_channel=feat_dim, n_emb_dims=feat_dim, num_layers=num_layers,
                            disable_contraction=disable_contraction)
        elif self.graph_backbone == 'gcn':
            self.gnn = SimpleGCN(in_channel=feat_dim, n_emb_dims=feat_dim, num_layers=num_layers)
        else:
            raise ValueError(f'Unsupported graph_backbone: {graph_backbone}')
        self.knn_k = knn_k
        self.topk_pool = int(topk_pool)
        self.attn_dropout = float(attn_dropout)
        self.fps_alpha = float(fps_alpha)
        self.drop_edge_prob = float(drop_edge_prob)
        self.feature_jitter = float(feature_jitter)
        
        #把原先简单的全连接改为更深的特征预测头，避免过拟合于局部纹理
        self.attn_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1)
        )
        
        # GNN 输出后的 Dropout 层
        self.gnn_dropout = nn.Dropout(p=gnn_dropout)
        
        # 初始化为 1.0，即起始时两者等权
        self.fusion_w = nn.Parameter(torch.ones(2))
        
        if use_ntn:
            self.scorer = InteractionScorer(
                feat_dim=feat_dim,
                tensor_neurons=tensor_neurons,
                dropout_rate=dropout_rate,
                ntn_temperature=ntn_temperature,
                l2_temperature=l2_temperature,
            )


    def forward(self, data, return_debug=False):
        full_img = data['full_img']   # [B, 1, H, W]
        # 处理 batch 维度，确保输入是 4D
        if full_img.dim() == 3:
            full_img = full_img.unsqueeze(0)
            
        coords_raw = data['coords']  # [N, 2] or [B, N, 2]
        if coords_raw.dim() == 2:
            coords = coords_raw.unsqueeze(0) # [1, N, 2]
        else:
            coords = coords_raw
        
        # 获取 node_mask
        node_mask = data.get('node_mask', None)
        if node_mask is not None and node_mask.dim() == 1:
            node_mask = node_mask.unsqueeze(0)  # [1, N]
        
        # 1. 节点基础特征提取
        h = self.encoder(full_img, coords) # [B*N, 128]
        
        B = full_img.size(0)
        N = coords.size(1)
        # 将 h 转换为 [B, feat_dim, N] 以输入 HGNN
        h = h.view(B, N, -1).transpose(1, 2)
        
        # Feature Jittering：训练时给节点特征添加微小的随机噪声
        if self.training and self.feature_jitter > 0:
            noise = torch.randn_like(h) * self.feature_jitter
            h = h + noise 
        
        # 2. 使用预计算的邻接矩阵 W
        W = data['W']  # [B, N, N] 或 [N, N]
        if W.dim() == 2:
            W = W.unsqueeze(0)  # [1, N, N]
        W = W.to(full_img.device)
        
        # DropEdge 机制：训练时随机切断图中的边
        if self.training and self.drop_edge_prob > 0:
            mask = (torch.rand_like(W) > self.drop_edge_prob).float()
            W = W * mask
        
        # 3. 运行图骨干网络（HGNN 或 GCN）
        if self.graph_backbone == 'hgnn':
            raw_H, H, edge_score, feat_final = self.gnn(h, W)
        else:
            feat_final, adjacency_for_topology = self.gnn(h, W)
            raw_H = adjacency_for_topology
            H = adjacency_for_topology
            edge_score = adjacency_for_topology
        feat_final = self.gnn_dropout(feat_final)
        node_feat = feat_final.transpose(1, 2)  # [B, N, feat_dim]

        # 4. Readout: 融合纹理 (Texture) 与拓扑 (Topology) 特征
        # texture_logits 由 MLP 从卷积特征中提取
        texture_logits = self.attn_head(node_feat).squeeze(-1) # [B, N]
        
        # topology_score 由图结构计算（GF-NMS 拉普拉斯显著性）
        # HGNN: 使用超图关联矩阵 H；GCN: 使用邻接矩阵作为近似结构输入
        topology_norm = compute_gf_score_from_H(H, remove_self_loop=True, use_log=True)
        
        # 应用 node_mask（忽略 padding 节点）
        if node_mask is not None:
            topology_norm = topology_norm.masked_fill(~node_mask, 0.0)

        # w[0] -> 纹理权重, w[1] -> 拓扑权重
        # w/o Topology_Score 消融时强制退化为纯纹理打分
        if self.disable_topology_score:
            logits = texture_logits
        else:
            w = torch.softmax(self.fusion_w, dim=0)
            logits = (w[0] * texture_logits + w[1] * topology_norm)
        
        # 应用温度缩放 (Temperature Scaling) 让注意力具有更平滑的探索能力
        temperature = 10.0
        logits = logits / temperature

        # 应用 node_mask 到 logits
        if node_mask is not None:
            logits = logits.masked_fill(~node_mask, -1e9)

        scores = torch.sigmoid(logits)  # [B, N] 概率分布
        importance_for_loss = logits  # 传递 Logits 以便计算 Focal Loss

        # 4. Readout: 加权 FPS (Weighted Farthest Point Sampling)
        
        # 预计算整个 batch 的距离矩阵 [B, N, N]
        dist_matrix_batch = torch.cdist(coords, coords)
        
        # 初始化：每个样本的 dist_to_selected, curr_logits, selected_idx
        dist_to_selected = torch.full((B, N), 10.0, device=node_feat.device)
        curr_logits = logits.clone()
        selected_mask = torch.zeros((B, N), dtype=torch.bool, device=node_feat.device)
        
        # 计算所有样本的 K_pool [B]
        if node_mask is not None:
            valid_nodes = node_mask.sum(dim=1)  # [B]
        else:
            valid_nodes = torch.full((B,), N, dtype=torch.long, device=node_feat.device)
        K_pool_batch = torch.clamp(valid_nodes, min=1, max=self.topk_pool)  # [B]
        max_K = self.topk_pool
        
        # 训练时应用采样 Dropout
        if self.training and self.attn_dropout > 0:
            mask = (torch.rand((B, N), device=curr_logits.device) > self.attn_dropout).float()
            if node_mask is not None:
                mask = mask * node_mask.float()
            curr_logits = curr_logits + (1.0 - mask) * -1e9
        
        # 应用 node_mask
        if node_mask is not None:
            curr_logits = curr_logits.masked_fill(~node_mask, -1e9)
        
        # 收集每个样本选中的索引（用 mask 表示）
        selected_batch_mask = torch.zeros((B, N), dtype=torch.bool, device=node_feat.device)
        
        # 串行迭代 K 次
        for k in range(max_K):
            # 只对还没选满 K_pool 个点的样本进行操作
            active = (k < K_pool_batch)  # [B]
            
            if not active.any():
                break
            
            # 计算 hybrid score [B, N]
            hybrid_score = curr_logits + self.fps_alpha * torch.log(dist_to_selected + 1e-6)
            
            # 对于已经选满的样本，设为 -inf
            hybrid_score[~active] = -1e9
            
            # 选择每个样本的下一个点 [B]
            idx = torch.argmax(hybrid_score, dim=1)  # [B]
            
            # 更新 selected_mask
            selected_batch_mask[torch.arange(B, device=node_feat.device), idx] = True
            
            # 更新 dist_to_selected：取到当前选中点的最小距离
            dist_to_selected = torch.min(
                dist_to_selected, 
                dist_matrix_batch[torch.arange(B, device=node_feat.device), idx, :]
            )
            
            # 将已选点的 logits 设为 -inf
            curr_logits[torch.arange(B, device=node_feat.device), idx] = -1e9
        
        # 收集所有选中的索引
        fingerprints = []
        selected_idx_batch = []
        for b in range(B):
            sample_selected_idx = torch.where(selected_batch_mask[b])[0]
            sample_selected_idx = sample_selected_idx[:K_pool_batch[b]]  # 确保不超过 K_pool
            
            topk_idx = sample_selected_idx
            selected_idx_batch.append(topk_idx)
            topk_scores = scores[b, topk_idx]
            topk_feat = node_feat[b, topk_idx]

            w_topk = topk_scores.unsqueeze(1)  # [K, 1]

            v_fingerprint = (topk_feat * w_topk).sum(dim=0) / (w_topk.sum(dim=0) + 1e-6)  # [feat_dim]
            fingerprints.append(v_fingerprint)
        
        # Stack 所有样本的 fingerprint
        v_fingerprint = torch.stack(fingerprints, dim=0)  # [B, feat_dim]
        
        # 统一训练和验证的特征归一化，释放欧式距离的学习空间
        v_fingerprint = F.normalize(v_fingerprint, p=2, dim=-1)
        
        # 对于 loss，返回所有节点的 logits（batch 形式）
        if return_debug:
            debug_info = {
                'scores': scores.detach().clone(),
                'logits': importance_for_loss.detach().clone(),
                'texture_logits': texture_logits.detach().clone(),
                'topology_norm': topology_norm.detach().clone(),
                'selected_batch_mask': selected_batch_mask.detach().clone(),
                'selected_idx': [idx.detach().clone() for idx in selected_idx_batch],
                'k_pool_batch': K_pool_batch.detach().clone(),
            }
            return v_fingerprint, importance_for_loss, debug_info

        return v_fingerprint, importance_for_loss

    def forward_baseline_b(self, data):
        # 1. 仅提取 CNN 特征 (h)
        full_img = data['full_img']
        if full_img.dim() == 3:
            full_img = full_img.unsqueeze(0)
        coords_raw = data['coords']
        if coords_raw.dim() == 2:
            coords = coords_raw.unsqueeze(0)
        else:
            coords = coords_raw
        
        h = self.encoder(full_img, coords)  # [B*N, 128]
        
        B = full_img.size(0)
        N = coords.size(1)
        # 将 h 转换为 [B, N, feat_dim]
        h = h.view(B, N, -1)
        
        # 2. 暴力平均 (Global Pooling)，完全忽略拓扑 H
        v_fingerprint = h.mean(dim=1)  # [B, feat_dim]
        v_fingerprint = F.normalize(v_fingerprint, p=2, dim=-1)
        
        # 3. 正常的 importance_for_loss (为了兼容 loss 脚本，可以传个全 0)
        importance_for_loss = torch.zeros_like(coords[:,:,0])  # [B, N]
        
        return v_fingerprint, importance_for_loss
