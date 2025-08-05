#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from itertools import zip_longest




class BaseGCN(nn.Module):
    def __init__(self, N=1024):
        super(BaseGCN, self).__init__()
        self.point_count = N

    def forward(self, x):
        raise NotImplementedError

    def construct_adjacency(self, point_features):
        batch_size, num_points, _ = point_features.size()
        # 点之间的成对距离
        pairwise_distances = torch.cdist(point_features, point_features, p=2)  
        # 创建邻接矩阵
        _, neighbor_indices = torch.topk(pairwise_distances, k=self.k, dim=-1, largest=False)  
        adj_matrices = torch.zeros(batch_size, num_points, num_points, device=point_features.device)
        adj_matrices.scatter_(2, neighbor_indices, 1)  
        return adj_matrices

    def normalize_adjacency(self, adj_matrices):
        batch_size, num_points, _ = adj_matrices.size()
        # 计算度矩阵
        node_degrees = torch.sum(adj_matrices, dim=2)  
        degree_matrix = torch.diag_embed(node_degrees)  
        degree_sqrt_inv = torch.sqrt(torch.reciprocal(degree_matrix))  
        degree_sqrt_inv[torch.isinf(degree_sqrt_inv)] = 0.0  

        identity = torch.eye(num_points, device=degree_matrix.device).unsqueeze(0).repeat(batch_size, 1, 1)
        normalized_matrices = identity - torch.bmm(torch.bmm(degree_sqrt_inv, adj_matrices), degree_sqrt_inv.transpose(1, 2))

        return normalized_matrices

    def add_self_loops_and_normalize(self, adj_matrices):
        batch_size, num_points, _ = adj_matrices.size()
        identity_matrix = torch.eye(num_points, device=adj_matrices.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # 添加自环
        adj_with_self_loops = adj_matrices + identity_matrix
        # 计算新的度矩阵
        degree_hat = torch.diag_embed(torch.sum(adj_with_self_loops, dim=2))
        degree_hat_sqrt_inv = torch.sqrt(torch.reciprocal(degree_hat))
        degree_hat_sqrt_inv[torch.isinf(degree_hat_sqrt_inv)] = 0.0

        normalized_adj = torch.bmm(torch.bmm(degree_hat_sqrt_inv, adj_with_self_loops), degree_hat_sqrt_inv)

        return normalized_adj

class BatchNormWrapper(nn.Module):
    def __init__(self, feature_dim):
        super(BatchNormWrapper, self).__init__()
        self.batch_norm = nn.BatchNorm1d(feature_dim)

    def forward(self, input_data):
        batch_size, num_points, feature_dim = input_data.size()
        # 重塑数据以适应BatchNorm1d
        flattened_data = input_data.view(-1, feature_dim)
        normalized = self.batch_norm(flattened_data)
        reshaped_output = normalized.view(batch_size, num_points, -1)
        return reshaped_output

class GCNResidualBlock(nn.Module):
    def __init__(self, args, in_features, hidden_dim, out_dim):
        super(GCNResidualBlock, self).__init__()
        self.gcn_layer1 = GCNLayer(args, in_features, hidden_dim)
        self.gcn_layer2 = GCNLayer(args, hidden_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.dropout_rate = args.dropout

    def forward(self, node_features, adjacency):
        # 第一层GCN
        hidden = self.gcn_layer1(node_features, adjacency)  
        hidden = F.relu(hidden)
        # 第二层GCN
        output = self.gcn_layer2(hidden, adjacency)  
        
        # 批量归一化
        batch_size, num_nodes, channels = output.size()
        reshaped = output.view(-1, channels)
        normalized = self.batch_norm(reshaped)  
        output = normalized.view(batch_size, num_nodes, channels)
        
        # 残差连接和激活
        return F.relu(output + node_features)

class GCNResNet(BaseGCN):
    def __init__(self, args, N=1024):
        super(GCNResNet, self).__init__()
        in_features, hidden_features, out_features, num_blocks = args.res_in, args.res_hid, args.res_out, args.res_num_blocks
        self.initial_gcn = GCNLayer(args, in_features, hidden_features)
        self.bn = nn.BatchNorm1d(hidden_features)
        self.res_blocks = nn.ModuleList([
            GCNResidualBlock(args, hidden_features, hidden_features, hidden_features)
            for _ in range(num_blocks)
        ])
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        self.final_gcn = GCNLayer(args, hidden_features, out_features)

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, 128)
        self.bn2 = BatchNormWrapper(features=128)

        self.linear3 = nn.Linear(128, 512)
        self.bn3 = BatchNormWrapper(features=512)

        self.linear4 = nn.Linear(512, 1024)
        self.bn4 = BatchNormWrapper(features=1024)

        self.linear5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.linear6 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.linear7 = nn.Linear(512, out_features)
        self.LinearLayers1 = nn.Sequential(self.linear2, self.bn2, nn.ReLU(), self.linear3, self.bn3, nn.ReLU(), self.linear4, self.bn4, nn.ReLU())
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.LinearLayers2 = nn.Sequential(self.linear5, self.bn5, nn.ReLU(), self.linear6, self.bn6, nn.ReLU(), self.linear7)

        
    def forward(self, x):
        adjs = self.construct_adjacency(x)

        if self.address_overfitting:
            A = self.add_self_loops_and_normalize(adjs).float()
        else:
            A = self.normalize_adjacency(adjs).float()

        h = self.initial_gcn(x, A)
        h = F.relu(h)
        B, N, C = h.size()
        h = h.view(-1, C)
        h = self.bn(h)
        h = h.view(B, N, C)
        
        for block in self.res_blocks:
            h = block(h, A)
        
        h = self.LinearLayers1(h)
        h = self.pool1(h.transpose(1, 2)).squeeze() # (B, N, out_features) -> (B, out_features)
        h = self.LinearLayers2(h)

        # h = self.final_gcn(h, A)
        return h

class GCNLayer(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = args.dropout
        # self.dp1 = nn.Dropout(p=args.dropout)

    def forward(self, x, adj):
        output = self.linear(x)
        output = F.relu(torch.bmm(adj, output))
        # output = self.dp1(output)
        return output

class GCN(BaseGCN):
    def __init__(self, args, in_c=3, hid_c=64, out_c=40, N=1024) -> None:
        super(GCN, self).__init__()
        N = args.num_points
        self.linear1 = nn.Linear(in_c, hid_c)
        self.linear2 = nn.Linear(hid_c, 128)
        self.bn2 = BatchNormWrapper(features=128)

        self.linear3 = nn.Linear(128, 512)
        self.bn3 = BatchNormWrapper(features=512)

        self.linear4 = nn.Linear(512, 1024)
        self.bn4 = BatchNormWrapper(features=1024)

        self.linear5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.linear6 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.linear7 = nn.Linear(512, out_c)
        self.gcn_layers = args.gcn_layers
        self.act = nn.ReLU()
        self.dropout = args.dropout
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        self.gc_layers = nn.ModuleList([GCNLayer(args, in_c, hid_c)])
        for _ in range(self.gcn_layers - 1):
            self.gc_layers.append(GCNLayer(args, hid_c, hid_c))
        self.LinearLayers1 = nn.Sequential(self.linear2, self.bn2, nn.ReLU(), self.linear3, self.bn3, nn.ReLU(), self.linear4, self.bn4, nn.ReLU())
        self.pool1 = nn.MaxPool1d(N)
        self.LinearLayers2 = nn.Sequential(self.linear5, self.bn5, nn.ReLU(), self.linear6, self.bn6, nn.ReLU(), self.linear7)
    
    def create_batchnorm(self, features):
        return lambda data: nn.BatchNorm1d(features)(data.view(-1, features)).view(data.size())

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        adjs = self.construct_adjacency(x)
        # ipdb.set_trace() 
        if self.address_overfitting:
            A = self.add_self_loops_and_normalize(adjs).float()
        else:
            A = self.normalize_adjacency(adjs).float()

        for layer in self.gc_layers:
            x = layer(x, A)
            x = F.dropout(x, self.dropout, training=self.training)
        # ipdb.set_trace()
        x = self.LinearLayers1(x)
        x = self.pool1(x.transpose(1, 2)).squeeze()
        x = self.LinearLayers2(x)
        # output = F.log_softmax(x, dim=-1)
        return x      

class MultiHeadAttention(nn.Module):
    def __init__(self, args, in_channels, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        # Q, K, V 投影矩阵
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        
        self.dropout = nn.Dropout(args.dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        
        # 投影并分头
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力权重
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权聚合
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # [B, N, C]
        out = self.out_proj(out)
        
        return out

class AttentionGCNLayer(nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(AttentionGCNLayer, self).__init__()
        self.gcn = GCNLayer(args, in_channels, out_channels)
        self.attention = MultiHeadAttention(args, out_channels)  # 使用输出通道数
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(args.dropout)
        
        # 添加输入特征转换
        self.input_transform = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, adj):
        # 转换输入特征维度以匹配GCN输出
        identity = self.input_transform(x)
        
        # GCN路径
        gcn_out = self.gcn(x, adj)
        x = self.norm1(identity + self.dropout(gcn_out))
        
        # 注意力路径
        attn_out = self.attention(x)
        x = self.norm2(x + self.dropout(attn_out))
        
        return x

class AttentionResidualBlock(nn.Module):
    def __init__(self, args, in_channels, hidden_dim, out_channels):
        super(AttentionResidualBlock, self).__init__()
        # 第一个 GCN+Attention 层
        self.attention_layer1 = AttentionGCNLayer(args, in_channels, hidden_dim)
        # 第二个 GCN+Attention 层
        self.attention_layer2 = AttentionGCNLayer(args, hidden_dim, out_channels)
        
        # 残差连接的特征变换（如果输入输出维度不同）
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
        
        # 归一化层
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, adj):
        identity = x
        
        # 两层 GCN+Attention
        out = self.attention_layer1(x, adj)
        out = F.relu(out)
        out = self.attention_layer2(out, adj)
        
        # 残差连接
        if hasattr(self.shortcut, 'weight'):
            identity = self.shortcut(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), -1)
        
        out = self.norm(out + identity)
        out = F.relu(out)
        return self.dropout(out)

class MultiHeadAttentionGCN(BaseGCN):
    def __init__(self, args, in_c=3, hid_c=64, out_c=40, N=1024) -> None:
        super(MultiHeadAttentionGCN, self).__init__()
        N = args.num_points
        
        # 基础参数
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        self.dropout = args.dropout
        
        # 初始特征变换
        self.input_transform = nn.Linear(in_c, hid_c)
        
        # 初始 GCN+Attention 层
        self.initial_layer = AttentionGCNLayer(args, in_c, hid_c)
        
        # 残差块
        num_blocks = args.res_num_blocks
        self.res_blocks = nn.ModuleList([
            AttentionResidualBlock(args, hid_c, hid_c, hid_c)
            for _ in range(num_blocks)
        ])
        
        # 特征提取层
        self.feature_layers = nn.Sequential(
            nn.Linear(hid_c, 128),
            BatchNormWrapper(features=128),
            nn.ReLU(),
            
            nn.Linear(128, 512),
            BatchNormWrapper(features=512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            BatchNormWrapper(features=1024),
            nn.ReLU()
        )
        
        # 全局池化
        self.pool = nn.MaxPool1d(N)
        
        # 分类层
        self.classification_layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(512, out_c)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 计算邻接矩阵
        adj = self.get_adj(x)
        if self.address_overfitting:
            adj = self.address_overfitting_graph(adj).float()
        else:
            adj = self.process_graph(adj).float()
        
        # 初始特征提取
        features = self.initial_layer(x, adj)
        
        # 通过残差块
        for res_block in self.res_blocks:
            features = res_block(features, adj)
        
        # 特征提取
        features = self.feature_layers(features)
        
        # 全局池化
        features = self.pool(features.transpose(1, 2)).reshape(batch_size, -1)
        
        # 分类
        output = self.classification_layers(features)
        
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultiHeadAttentionGCN")
        parser.add_argument('--num_blocks', type=int, default=3,
                          help='Number of residual blocks')
        parser.add_argument('--num_heads', type=int, default=8,
                          help='Number of attention heads')
        return parent_parser

class DenseGCNBlock(nn.Module):
    def __init__(self, args, in_features, growth_rate):
        super(DenseGCNBlock, self).__init__()
        self.gcn = GCNLayer(args, in_features, growth_rate)
        self.bn = nn.BatchNorm1d(growth_rate)
        self.dropout = args.dropout

    def forward(self, x, adj):
        out = self.gcn(x, adj)
        B, N, C = out.size()
        out = out.view(-1, C)
        out = self.bn(out)
        out = out.view(B, N, C)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        return out

class DenseGCNLayer(nn.Module):
    def __init__(self, args, num_layers, in_features, growth_rate):
        super(DenseGCNLayer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseGCNBlock(args, in_features + i * growth_rate, growth_rate))
        
        self.out_channels = in_features + num_layers * growth_rate

    def forward(self, x, adj):
        features = [x]
        for layer in self.layers:
            # 连接所有之前的特征
            inputs = torch.cat(features, dim=-1)
            new_features = layer(inputs, adj)
            features.append(new_features)
        return torch.cat(features, dim=-1)

class TransitionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.conv = nn.Linear(in_features, out_features)

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(-1, C)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x.view(B, N, -1)
        return x

class DenseResGCN(BaseGCN):
    def __init__(self, args, N=1024):
        super(DenseResGCN, self).__init__()
        self.num_points = args.num_points
        
        # 网络参数
        in_features = args.res_in
        hidden_features = args.res_hid
        out_features = args.res_out
        growth_rate = args.growth_rate if hasattr(args, 'growth_rate') else 32
        block_layers = args.block_layers if hasattr(args, 'block_layers') else 4
        num_blocks = args.res_num_blocks
        
        # 基础参数
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        
        # 初始卷积
        self.initial_gcn = GCNLayer(args, in_features, hidden_features)
        self.initial_bn = nn.BatchNorm1d(hidden_features)
        
        # Dense块和转换层
        self.dense_layers = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        
        current_features = hidden_features
        for i in range(num_blocks):
            # 添加Dense块
            dense_layer = DenseGCNLayer(args, block_layers, current_features, growth_rate)
            self.dense_layers.append(dense_layer)
            current_features = dense_layer.out_channels
            
            # 添加转换层（除最后一个块）
            if i != num_blocks - 1:
                trans_layer = TransitionLayer(current_features, current_features // 2)
                self.trans_layers.append(trans_layer)
                current_features = current_features // 2
        
        # 特征提取层
        self.linear1 = nn.Linear(current_features, 128)
        self.bn1 = BatchNormWrapper(features=128)

        self.linear2 = nn.Linear(128, 512)
        self.bn2 = BatchNormWrapper(features=512)

        self.linear3 = nn.Linear(512, 1024)
        self.bn3 = BatchNormWrapper(features=1024)
        
        # 分类层
        self.linear4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)

        self.linear5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.linear6 = nn.Linear(512, out_features)
        
        # Sequential层
        self.feature_layers = nn.Sequential(
            self.linear1, self.bn1, nn.ReLU(),
            self.linear2, self.bn2, nn.ReLU(),
            self.linear3, self.bn3, nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            self.linear4, self.bn4, nn.ReLU(),
            self.linear5, self.bn5, nn.ReLU(),
            self.linear6
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 计算邻接矩阵
        adj = self.get_adj(x)
        if self.address_overfitting:
            adj = self.address_overfitting_graph(adj).float()
        else:
            adj = self.process_graph(adj).float()
        
        # 初始特征提取
        features = self.initial_gcn(x, adj)
        B, N, C = features.size()
        features = features.view(-1, C)
        features = self.initial_bn(features)
        features = features.view(B, N, C)
        features = F.relu(features)
        
        # Dense块处理
        for i, (dense_layer, trans_layer) in enumerate(zip_longest(
            self.dense_layers, self.trans_layers)):
            # Dense层
            features = dense_layer(features, adj)
            # 转换层（除最后一个块）
            if trans_layer is not None:
                features = trans_layer(features)
        
        # 特征提取
        features = self.feature_layers(features)
        
        # 全局池化
        features = self.pool(features.transpose(1, 2)).squeeze(-1)
        
        # 分类
        output = self.classifier(features)
        
        return output

class DilatedGCNLayer(nn.Module):
    def __init__(self, args, in_channels, out_channels, dilation):
        super(DilatedGCNLayer, self).__init__()
        self.feature_transform = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = args.dropout
        self.dilation = dilation
        self.k = args.k

    def forward(self, x, adj):
        # 特征变换
        out = self.feature_transform(x)
        # GCN卷积
        out = torch.bmm(adj, out)
        # 批归一化
        B, N, C = out.size()
        out = out.view(-1, C)
        out = self.bn(out)
        out = out.view(B, N, C)
        # 激活和dropout
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        return out

class DilatedGCN(BaseGCN):
    def __init__(self, args, in_c=3, hid_c=64, out_c=40, N=1024) -> None:
        super(DilatedGCN, self).__init__()
        self.num_points = args.num_points
        
        # 基础参数
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        self.dropout = args.dropout
        
        # 初始特征变换
        self.input_transform = nn.Linear(in_c, hid_c)
        
        # 空洞GCN层
        self.gcn_layers = nn.ModuleList()
        num_layers = args.gcn_layers
        
        # 第一层使用标准卷积
        self.gcn_layers.append(DilatedGCNLayer(args, in_c, hid_c, dilation=1))
        
        # 后续层使用递增的空洞率
        for i in range(1, num_layers):
            dilation = 2 ** i  # 空洞率: 1, 2, 4, 8, ...
            self.gcn_layers.append(DilatedGCNLayer(args, hid_c, hid_c, dilation))
        
        # 特征提取层
        self.linear1 = nn.Linear(hid_c, 128)
        self.bn1 = BatchNormWrapper(features=128)
        
        self.linear2 = nn.Linear(128, 512)
        self.bn2 = BatchNormWrapper(features=512)
        
        self.linear3 = nn.Linear(512, 1024)
        self.bn3 = BatchNormWrapper(features=1024)
        
        # 分类层
        self.linear4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        
        self.linear5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        
        self.linear6 = nn.Linear(512, out_c)
        
        # Sequential层
        self.feature_layers = nn.Sequential(
            self.linear1, self.bn1, nn.ReLU(),
            self.linear2, self.bn2, nn.ReLU(),
            self.linear3, self.bn3, nn.ReLU()
        )
        
        self.pool = nn.MaxPool1d(N)
        
        self.classifier = nn.Sequential(
            self.linear4, self.bn4, nn.ReLU(),
            self.linear5, self.bn5, nn.ReLU(),
            self.linear6
        )

    def compute_dilated_adjacency(self, x, dilation):
        batch_size, num_points, _ = x.size()
        
        # 计算点对之间的距离
        distances = torch.cdist(x, x, p=2)
        
        # 对每个点选择k个最近邻，考虑空洞率
        dilated_k = min(self.k * dilation, num_points - 1)
        _, neighbor_indices = torch.topk(distances, k=dilated_k + 1, dim=-1, largest=False)
        
        # 按照空洞率采样邻居
        sampled_indices = neighbor_indices[:, :, 1::dilation]  # 跳过自身连接
        sampled_indices = sampled_indices[:, :, :self.k]  # 保持每个点的邻居数量一致
        
        # 构建邻接矩阵
        adj_matrix = torch.zeros(batch_size, num_points, num_points, device=x.device)
        adj_matrix.scatter_(2, sampled_indices, 1)
        
        return adj_matrix

    def forward(self, x):
        batch_size = x.shape[0]
        features = x
        
        # 多尺度特征聚合
        multi_scale_features = []
        
        # 通过不同空洞率的GCN层
        for gcn_layer in self.gcn_layers:
            # 计算当前层的空洞邻接矩阵
            dilated_adj = self.compute_dilated_adjacency(features, gcn_layer.dilation)
            
            # 应用归一化
            if self.address_overfitting:
                dilated_adj = self.address_overfitting_graph(dilated_adj).float()
            else:
                dilated_adj = self.process_graph(dilated_adj).float()
            
            # 应用GCN层
            features = gcn_layer(features, dilated_adj)
            multi_scale_features.append(features)
        
        # 融合多尺度特征
        features = torch.stack(multi_scale_features, dim=-1).mean(dim=-1)
        
        # 特征提取
        features = self.feature_layers(features)
        
        # 全局池化
        features = self.pool(features.transpose(1, 2)).reshape(batch_size, -1)
        
        # 分类
        output = self.classifier(features)
        
        return output

    def get_receptive_field(self):
        receptive_field = self.k
        for layer in self.gcn_layers[1:]:
            receptive_field += (self.k - 1) * layer.dilation
        return receptive_field


