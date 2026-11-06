# Portions derived from HyperGCT, Copyright (c) 2024 zhangxy0517.
# MIT license; see licenses/HyperGCT-LICENSE.txt and THIRD_PARTY_NOTICES.md.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def distance(x):  # bs, channel, num_points
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = xx + inner + xx.transpose(2, 1).contiguous()  # bs, num_points, num_points
    return distance


def feature_knn(x, k):
    dis = -distance(x)
    idx = dis.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def mask_score(score, H0, iter, num_layer, choice='topk'):
    bs, num, _ = H0.size()
    # score: row->V, col->E
    if choice == 'auto':
        W = score * H0
        H = torch.where(torch.greater(W, 0), H0, torch.zeros_like(H0))

    elif choice == 'topk':
        k = round(num * 0.1 * (num_layer - 1 - iter))
        topk, _ = torch.topk(score, k=k, dim=-1)  # 每个节点选出topk概率的超边，该超边包含该节点
        a_min = torch.min(topk, dim=-1).values.unsqueeze(-1).repeat(1, 1, num)
        W = torch.where(torch.greater_equal(score, a_min), score, torch.zeros_like(score))
        H = torch.where(torch.greater(W, 0), H0, torch.zeros_like(H0))

    else:
        raise NotImplementedError

    return H, W


class GraphUpdate(nn.Module):
    def __init__(self, num_channels, num_heads, num_layer):
        super(GraphUpdate, self).__init__()
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads
        self.num_layer = num_layer
        self.make_score_choice = 'topk'

    def forward(self, H0, vertex_feat, edge_feat, iter):
        # 根据 vertex_feat 和 edge_feat 更新 H （attention）, 计算相关参数: 决定下一轮中边的特征从哪些点得到
        # vertex_feat: [B, num_channels, N], edge_feat: [B, num_channels, N]
        bs, num_channels, num_vertices = vertex_feat.size()
        
        # 需要转置为 [B, N, num_channels] 以便正确 reshape 为 [B, head, num_channels//head, N]
        vertex_feat_t = vertex_feat.transpose(1, 2)  # [B, N, num_channels]
        edge_feat_t = edge_feat.transpose(1, 2)  # [B, N, num_channels]
        
        Q = self.projection_q(vertex_feat).view([bs, self.head, num_channels // self.head, num_vertices])  # row
        K = self.projection_k(edge_feat).view([bs, self.head, num_channels // self.head, num_vertices])  # col

        attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5  # [bs,
        # head, num_corr, num_corr]



        # mask attention using SC2 prior
        attention_mask = 1 - H0  # 固定图结构，验证主网络
        attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e9)
        score = torch.sigmoid(attention + attention_mask[:, None, :,
                                          :])  # [bs, head, num_corr, num_corr] row->V, col->E, 考察每个节点分别属于每个边的概率(col上的softmax,因为每个节点经过mask后属于的边数不同)
        # if V is not None:
        #     edge_message = torch.einsum('bhoi, bhci-> bhco', score, V).reshape([bs, -1, num_vertices])  # [bs, dim, num_corr]

        score = (torch.sum(score, dim=1) / self.head).view(bs, num_vertices, num_vertices)  # 合并多个head的score
        H, W = mask_score(score, H0, iter, self.num_layer, choice=self.make_score_choice)

        # update D_n_1, W_edge according to new H
        degree_E = H.sum(dim=1)
        De = torch.diag_embed(degree_E)  # torch.sparse_matrix
        De_n_1 = De ** -1
        De_n_1[torch.isinf(De_n_1)] = 0

        degree_V = H.sum(dim=2)
        Dv = torch.diag_embed(degree_V)
        Dv_n_1 = Dv ** -1
        Dv_n_1[torch.isinf(Dv_n_1)] = 0

        W_edge = W.sum(dim=1)  # [bs, num]
        W_edge = F.normalize(W_edge, dim=1)
        W_edge = W_edge.view([bs, num_vertices, 1])
        return H, W, De_n_1, Dv_n_1, W_edge


class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention, H):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5  # [bs,
        # head, num_corr, num_corr]
        attention_mask = 1 - H
        attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e9)

        # combine the feature similarity with spatial consistency
        weight = torch.softmax(attention[:, None, :, :] * feat_attention + attention_mask[:, None, :, :],
                               dim=-1)  #[bs, head, num_corr, num_corr]
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])  # [bs, dim, num_corr]
        message = self.fc_message(message)
        res = feat + message
        return res


class feature_aggregation_layer(nn.Module):
    def __init__(self,
                 k=20,
                 num_channels=128,
                 head=1,
                 use_knn=False,
                 aggr='mean'):
        super(feature_aggregation_layer, self).__init__()
        self.k = k
        self.use_knn = use_knn
        self.head = head
        self.num_channels = num_channels
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Conv1d(2 * num_channels, num_channels, kernel_size=1),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        if self.aggr == 'self_attention':
            self.fc = nn.Sequential(
                nn.Conv1d(num_channels, num_channels // 4, kernel_size=1),
                nn.BatchNorm1d(num_channels // 4),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Conv1d(num_channels // 4, 1, kernel_size=1),
            )

    def forward(self, x, y, W, H, De_n_1, Dv_n_1, W_edge, k):
        bs, num_dims, num_points = x.size()
        edge_feature = None
        if self.use_knn:
            corr_pos = x  # (bs, dim, num_corr)
            idx = feature_knn(corr_pos, k=k + 1)  # (bs, num_corr, k)
            idx = idx[:, :, 1:]  # ignore the center point

            # bs, num_points, _ = idx.size()
            idx_base = torch.arange(bs).to(x.device).view(-1, 1, 1) * num_points  #
            idx = idx + idx_base  #

            idx = idx.view(-1)
            # _, num_dims, _ = corr_pos.size()

            corr_pos = corr_pos.transpose(2, 1).contiguous()
            neighbor_corr_pose = corr_pos.view(bs * num_points, -1)[idx, :]
            neighbor_corr_pose = neighbor_corr_pose.view(bs, num_points, k, num_dims)
            corr_pos = corr_pos.view(bs, num_points, 1, num_dims)

            neighbor_corr_pose = (neighbor_corr_pose - corr_pos).permute(0, 3, 1, 2)

            neighbor_corr_pose = neighbor_corr_pose.max(dim=-1, keepdim=True)[
                0]  # attention here [bs, num_dim, num_point, k] -> [bs, num_dim, num_point, 1]

            neighbor_corr_pose = neighbor_corr_pose.view(bs, num_dims, num_points)
            feature = x + neighbor_corr_pose
            # feature = torch.cat((x, neighbor_corr_pose), dim=1)  # bs, dim*2, num_corr, k

        else:
            # feature aggregation using softmax or more complicate ways (attention)
            if self.aggr == 'mean':
                # v->e
                # aggregation message from v to e
                feature = x.permute(0, 2, 1)  # [bs, num, dim]
                feature = torch.bmm(H.permute(0, 2, 1), feature)
                edge_feature = torch.bmm(De_n_1, feature)

                if y is not None:
                    edge_feature = self.mlp(torch.cat((y, edge_feature.permute(0, 2, 1)), dim=1)).permute(0, 2,
                                                                                                          1)  # [bs, num, dim]

                # update message of e
                feature = W_edge * edge_feature  # [bs, num, 1] * [bs, num, dim]

                # aggregation message from e to v
                feature = torch.bmm(H, feature)
                feature = torch.bmm(Dv_n_1, feature)

                feature = feature.permute(0, 2, 1)  # [bs, dim, num]
                edge_feature = edge_feature.permute(0, 2, 1)

            elif self.aggr == 'self_attention':
                # e->v nonlocal net using weight matrix W
                feature = x.permute(0, 2, 1)
                feature = torch.bmm(H.permute(0, 2, 1), feature)  # H^T=H
                edge_feature = torch.bmm(De_n_1, feature)  # [bs, num, dim]

                feat = edge_feature.permute(0, 2, 1)  # [bs, dim, num]
                score = torch.softmax(self.fc(feat).permute(0, 2, 1), dim=1)  # [bs, num, 1]
                feature = score * W_edge * feature

                feature = torch.bmm(H, feature)
                feature = torch.bmm(Dv_n_1, feature)

                feature = feature.permute(0, 2, 1)
                edge_feature = edge_feature.permute(0, 2, 1)

            else:
                raise NotImplementedError

        return feature, edge_feature


class HGNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, residual_connection=False, use_edge_feature=True):
        super(HGNN_layer, self).__init__()
        self.con = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.fg = feature_aggregation_layer(k=20, num_channels=out_channels)
        self.residual_connection = residual_connection
        self.use_edge_feature = use_edge_feature

    def forward(self, x, y, W, H, De_n_1, Dv_n_1, W_edge, alpha, thete, x0, k):
        bs, num_vertex, num_edge = H.size()
        x1, y = self.fg(x, y, W, H, De_n_1, Dv_n_1, W_edge,
                        k)  # x1: feature of vertex in n+1 layer; y: feature of edge in n layer
        # update feature of vertex
        if self.residual_connection:
            x = x1 * (1 - alpha) + alpha * x0
            x = (1 - thete) * x + thete * self.bn(self.con(x)) 
        else:  
            x = self.bn(self.con(x1)) + x
        x = F.leaky_relu(x, negative_slope=0.2)
        if not self.use_edge_feature:
            y = None
        return x, y


class HGNN(nn.Module):
    def __init__(self, in_channel=6,
                 n_emb_dims=128,
                 k=20,
                 num_layers=6,
                 lamda=0.5,
                 alpha=0.1,
                 disable_contraction=False):
        super(HGNN, self).__init__()
        self.k = k
        self.lamda = lamda
        self.alpha = alpha
        self.num_layers = num_layers
        self.change_H0 = False
        # Ablation: when True, keep the high-order hypergraph backbone but DISABLE
        # dynamic contraction — H stays = H^0 (raw_H) for every layer. Isolates the
        # contribution of contraction from that of the high-order HGNN backbone.
        self.disable_contraction = bool(disable_contraction)
        dim = [n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims,
               n_emb_dims]  #  deeper network is not equal to better performance
        self.layer0 = nn.Conv1d(in_channel, dim[0], kernel_size=1, bias=True)
        self.blocks = nn.ModuleDict()
        for i in range(num_layers):
            self.blocks[f'GNN_layer_{i}'] = HGNN_layer(in_channels=dim[i], out_channels=dim[i + 1])
            self.blocks[f'NonLocal_{i}'] = NonLocalBlock(dim[i + 1])
            if i < num_layers - 1 and not self.disable_contraction:
                self.blocks[f'update_graph_{i}'] = GraphUpdate(num_channels=n_emb_dims, num_heads=1, num_layer=self.num_layers)

    def forward(self, x, W):
        """
        前向传播，支持 node_mask 以忽略 padding 节点。
        
        Args:
            x: [B, D, N] 节点特征
            W: [B, N, N] 邻接矩阵
        """
        global edge_score
        batch_size, num_dims, num_points = x.size()  # bs, 12, num_corr

        # 1. W[W>0] = 1
        H = W
        H[H > 0] = 1.0
        
        # 2. Degree of V, E.
        degree = H.sum(dim=1)

        raw_H = H
        D = torch.diag_embed(degree)  # torch.sparse_matrix
        De_n_1 = D ** -1
        De_n_1[torch.isinf(De_n_1)] = 0
        Dv_n_1 = De_n_1  # 初始超图矩阵 H0 是对称的

        # 3. edge weight = sum(W, dim=0)
        W_edge = W.sum(dim=1)  # [bs, num]
        W_edge = F.normalize(W_edge, p=2, dim=1)
        W_edge = W_edge.view([batch_size, num_points, 1])

        feat = self.layer0(x)
        feat0 = feat
        edge_feat = None
        edge_score = raw_H  # ensure defined even when contraction is disabled (static-H ablation)
        for i in range(self.num_layers):
            theta = math.log(self.lamda / (i + 1) + 1)
            feat, edge_feat = self.blocks[f'GNN_layer_{i}'](feat, edge_feat, W, H, De_n_1, Dv_n_1, W_edge, self.alpha,
                                                            theta, feat0, self.k)
            feat = self.blocks[f'NonLocal_{i}'](feat, W, raw_H)

            feat = F.normalize(feat, p=2, dim=1)
            edge_feat = F.normalize(edge_feat, p=2, dim=1)
            # change hypergraph dynamically
            if i < self.num_layers - 1 and not self.disable_contraction:
                H, edge_score, De_n_1, Dv_n_1, W_edge = self.blocks[f'update_graph_{i}'](H, feat, edge_feat,
                                                                                         i)
                #edge_score_list.append(edge_score)
                #H_list.append(H)

        return raw_H, H, edge_score, feat
