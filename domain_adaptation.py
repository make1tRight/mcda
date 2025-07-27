import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_manager import LossManager
import pandas as pd
import numpy as np
# import torch.optim as optim
from torch.autograd import Function
import warnings
from logger import *
from torch.utils.data import DataLoader
from config import Config
from typing import Optional
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
    category=UserWarning,
)


# ------------------------------
# 模块一：特征提取器
# ------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            # 备份
            # nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)

class DomainDiscriminator(nn.Module):
    """
    域鉴别器
    """
    def __init__(self, in_features):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Dropout(0.5))
    # def __init__(self, in_features):
    #     super(DomainDiscriminator, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(in_features, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 16),
    #         nn.ReLU(),
    #         nn.Linear(16, 2)
    #         # nn.Sigmoid(),
    #         # nn.Linear(in_features, 128),
    #         # nn.ReLU(),
    #         # nn.Linear(128, 1),
    #         # nn.Sigmoid()
    #     )

    def forward(self, x):
        return self.net(x)

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, 
                      out_channels=256, 
                      kernel_size=1, 
                      stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, 
                      out_channels=128, 
                      kernel_size=1, 
                      stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, 
                      out_channels=128, 
                      kernel_size=1, 
                      stride=1),
            nn.ReLU(),
            nn.Dropout(0.2))

    def forward(self, x):
        features = self.encoder(x)
        features = features.squeeze(-1)  # 移除最后一个维度，变为 (batch_size, 128)
        return features


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=128, 
                 n_heads=4,
                 n_layers=2, 
                 dropout=0.1):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 1. 线性映射到 embedding 维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)  # LayerNorm增加稳定性

        # 2. 可选的位置编码（若无序列结构，可以省略）
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # 3. Transformer 编码器堆叠
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # 提前归一化增加稳定性
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        # x: (batch_size, input_dim) → (batch_size, 1, hidden_dim)
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input after squeeze, got {x.dim()}D")
        x = self.input_proj(x)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("After input_proj: NaN or Inf")

        x = self.input_norm(x)
        x = x.unsqueeze(1) + self.pos_embedding
        # Transformer 编码
        x = self.encoder(x)  # (batch_size, 1, hidden_dim)
        x = x.squeeze(1)  # (batch_size, hidden_dim, 1)
        x = self.out_proj(x)  # (batch_size, hidden_dim)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=128,
                 n_heads=4,
                 n_layers=2,
                 dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.transformer = TransformerFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout)
    
    def forward(self, x):
        return self.transformer(x)


class ChannelAttentionBlock(nn.Module):
    """
    通道注意力块：专用于1D数据
    """
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttentionBlock, self).__init__()
        self.input_channels = input_channels
        self.reduction_ratio = reduction_ratio
        self.mid_channels = max(input_channels // reduction_ratio, 1)
        # 1. Average pool1d 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 2. fc
        self.fc1 = nn.Linear(input_channels, self.mid_channels)
        self.bn1 = nn.BatchNorm1d(self.mid_channels)
        self.dropout1 = nn.Dropout(0.1)
        # 3. fc
        self.fc2 = nn.Linear(self.mid_channels, input_channels)
        self.bn2 = nn.BatchNorm1d(input_channels)
        self.dropout2 = nn.Dropout(0.1)
        # 4. Sigmoid
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        # 1. Average pool1d
        pooled = self.avg_pool(x)
        # 2. reshape
        flattened = pooled.squeeze(-1)
        # 3. fc
        fc1_out = self.fc1(flattened)
        fc1_out = self.bn1(fc1_out)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        # 4. fc
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.bn2(fc2_out)
        fc2_out = self.dropout2(fc2_out)

        # 5. sigmoid
        attention_weights = self.sigmoid(fc2_out)
        # 6. 应用注意力权重
        attention_weights = attention_weights.unsqueeze(-1)
        # 应用注意力
        return identity * attention_weights + identity * 0.1


class FeatureEncoderWithAttention(nn.Module):
    """
    带注意力机制的特征编码器
    """
    def __init__(self, input_dim, use_attention=True, attention_type="channel"):
        super(FeatureEncoderWithAttention, self).__init__()

        self.use_attention = use_attention
        self.attention_type = attention_type
        # 基础编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, 
                      out_channels=256, 
                      kernel_size=1, 
                      stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, 
                      out_channels=128, 
                      kernel_size=1, 
                      stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, 
                      out_channels=128, 
                      kernel_size=1, 
                      stride=1),
            nn.ReLU(),
            nn.Dropout(0.2))
        # 注意力块
        if use_attention:
            if attention_type == "channel":
                self.attention = ChannelAttentionBlock(128)
            else:
                raise ValueError(f"Unsupported attention_type: {attention_type}")

    def forward(self, x):
        features = self.encoder(x)
        if self.use_attention:
            if self.attention_type == "channel":
                features = self.attention(features)

        features = features.squeeze(-1)  # 移除最后一个维度，变为 (batch_size, 128)
        return features


class Classifier(nn.Module):
    """
    模块二：分类器
    """
    def __init__(self, 
                 feature_dim_1, 
                 feature_dim_2, 
                 feature_dim_3, 
                 num_classes):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim_1, 
                      feature_dim_2),
            nn.ReLU(),
            nn.Linear(feature_dim_2, 
                      feature_dim_3),
            nn.ReLU(),
            nn.Linear(feature_dim_3, 
                      num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
# 模块三：域判别器
# ------------------------------
# 定义梯度反转层
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if torch.isnan(grad_output).any():
            print("grad_output contains NaN.")
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)

# def mcd_train(
#     source_loader: DataLoader,
#     target_loader: DataLoader,
#     model_e, model_c1, model_c2, model_d,
#     optimizer_e, optimizer_c1, optimizer_c2, optimizer_d,
#     cfg: Config,
#     logger=None,
#     loss_mgr=None,
#     lambda_GRL=1.0):
#     if loss_mgr is None:
#         loss_mgr = LossManager()
#     if logger is None:
#         logger = Logger(log_dir="./logs", level="INFO")
#     ce_loss = nn.CrossEntropyLoss()
#     # disc_loss = nn.MSELoss()
#     bce_loss = nn.BCEWithLogitsLoss()


#     # Step 1: 用源域有标签样本训练 E+C1+C2
#     model_e.train(); model_c1.train(); model_c2.train()
#     source_iter = iter(source_loader)
#     for step in range(cfg.step1_iter):
#         try:
#             source_batch = next(source_iter)
#         except StopIteration:
#             source_iter = iter(source_loader)
#             source_batch = next(source_iter)
#         x, y = source_batch
#         x, y = x.to(cfg.device), y.to(cfg.device)
#         features = model_e(x)
#         p1, p2 = model_c1(features), model_c2(features)

#         if torch.isnan(p1).any() or torch.isinf(p1).any():
#             logger.log("out1 contains NaN or Inf")
#         if torch.isnan(p2).any() or torch.isinf(p2).any():
#             logger.log("out2 contains NaN or Inf")

#         loss = ce_loss(p1, y) + ce_loss(p2, y)
#         optimizer_e.zero_grad(); optimizer_c1.zero_grad(); optimizer_c2.zero_grad()
#         loss.backward()
#         if cfg.clip_grad:
#             torch.nn.utils.clip_grad_norm_(
#                 list(model_e.parameters()) +
#                 list(model_c1.parameters()) +
#                 list(model_c2.parameters()), 
#                 cfg.clip_grad)
#         optimizer_e.step(); optimizer_c1.step(); optimizer_c2.step()
#         if (step + 1) % 100 == 0 or step == 0 or (step + 1) == cfg.step1_iter:
#             with torch.no_grad():
#                 pred = (p1 + p2) / 2
#                 pred_list = torch.bincount(
#                     pred.argmax(dim=1), minlength=loss_mgr.num_classes)
#                 logger.log("[step 1 - Iter {}/{}]", step + 1, cfg.step1_iter)
#                 logger.log("step 1 loss: {: .4f}", loss.item())
#                 logger.log("pred distribution: {}", pred_list.cpu().numpy())

#     # Step 2: 固定E，最大化目标域预测差异
#     model_e.eval(); model_c1.train(); model_c2.train()
#     target_iter = iter(target_loader)
#     for step in range(cfg.step2_iter):
#         try:
#             target_batch = next(target_iter)
#         except StopIteration:
#             target_iter = iter(target_loader)
#             target_batch = next(target_iter)
#         targetx = target_batch[0].to(cfg.device)
#         target_features = model_e(targetx).detach()
#         p1, p2 = model_c1(target_features), model_c2(target_features)
#         loss = -loss_mgr.discrepancy(torch.softmax(p1, 1), torch.softmax(p2, 1))
#         optimizer_c1.zero_grad(); optimizer_c2.zero_grad()
#         loss.backward()
#         optimizer_c1.step(); optimizer_c2.step()

#         if (step + 1) % 100 == 0 or step == 0 or (step + 1) == cfg.step2_iter:
#             # 打印 loss 和预测分布
#             with torch.no_grad():
#                 pred1 = p1.argmax(dim=1)
#                 pred2 = p2.argmax(dim=1)
#                 pred1_dist = torch.bincount(pred1, minlength=loss_mgr.num_classes)
#                 pred2_dist = torch.bincount(pred2, minlength=loss_mgr.num_classes)
#                 logger.log("[step 2 - Iter {}/{}]", step + 1, cfg.step2_iter)
#                 logger.log("loss: {: .4f}", loss.item())
#                 logger.log("C1 pred distribution: {}", pred1_dist.cpu().numpy())
#                 logger.log("C2 pred distribution: {}", pred2_dist.cpu().numpy())

#     # Step 3: 固定C1/C2，更新E最小化预测差异
#     model_e.train(); model_c1.eval(); model_c2.eval()
#     target_iter = iter(target_loader)
#     for step in range(cfg.step3_iter):
#         try:
#             target_batch = next(target_iter)
#         except StopIteration:
#             target_iter = iter(target_loader)
#             target_batch = next(target_iter)
#         targetx = target_batch[0].to(cfg.device)
#         target_features = model_e(targetx)
#         p1, p2 = model_c1(target_features), model_c2(target_features)
#         loss = loss_mgr.discrepancy(torch.softmax(p1, 1), torch.softmax(p2, 1))
#         optimizer_e.zero_grad(); loss.backward(); optimizer_e.step()
#         if (step + 1) % 100 == 0 or step == 0 or (step + 1) == cfg.step3_iter:
#             # 打印 loss 和预测分布
#             with torch.no_grad():
#                 pred = (p1 + p2) / 2
#                 pred_labels = pred.argmax(dim=1)
#                 pred_dist = torch.bincount(pred_labels, minlength=loss_mgr.num_classes)
#                 logger.log("[step 3 - Iter {}/{}]", step + 1, cfg.step3_iter)
#                 logger.log("loss: {: .4f}", loss.item())
#                 logger.log("pred distribution: {}", pred_dist.cpu().numpy())

#     # Step 4: 领域对抗训练
#     model_e.train(); model_d.train()
#     for step in range(cfg.step4_iter):
#         # 获取源域和目标域数据
#         try:
#             source_batch = next(source_iter)
#         except StopIteration:
#             source_iter = iter(source_loader)
#             source_batch = next(source_iter)
#         try:
#             target_batch = next(target_iter)
#         except StopIteration:
#             target_iter = iter(target_loader)
#             target_batch = next(target_iter)
#         sourcex = source_batch[0].to(cfg.device)
#         targetx = target_batch[0].to(cfg.device)
#         # 特征提取
#         source_features = model_e(sourcex)
#         target_features = model_e(targetx)
#         # 合并特征
#         features = torch.cat([source_features, target_features], dim=0)
#         # 领域标签
#         domain_labels = torch.cat([
#             torch.zeros(source_features.size(0), 1, device=cfg.device),
#             torch.ones(target_features.size(0), 1, device=cfg.device)], dim=0)
#         # 梯度反转
#         features_GRL = grad_reverse(features, lam=lambda_GRL)
#         domain_preds = model_d(features_GRL)
#         loss = bce_loss(domain_preds, domain_labels)
#         optimizer_d.zero_grad(); optimizer_e.zero_grad()
#         loss.backward()
#         optimizer_d.step(); optimizer_e.step()
#         if (step + 1) % 100 == 0 or step == 0 or (step + 1) == cfg.step4_iter:
#             logger.log("[step 4 - Iter {}/{}]", step + 1, cfg.step4_iter)
#             logger.log("loss: {: .4f}", loss.item())


# def mcd_train(
#     source_loader: DataLoader,
#     target_loader: DataLoader,
#     model_e, model_c1, model_c2, model_d,
#     optimizer_e, optimizer_c1, optimizer_c2, optimizer_d,
#     cfg: Config,
#     logger=None,
#     loss_mgr=None,
#     lambda_GRL=1.0,
#     num_epochs=80,
#     num_k=4  # Step3循环次数，论文推荐4
# ):
#     if loss_mgr is None:
#         loss_mgr = LossManager()
#     if logger is None:
#         logger = Logger(log_dir="./logs", level="INFO")
#     ce_loss = nn.CrossEntropyLoss()
#     bce_loss = nn.BCEWithLogitsLoss()

#     for epoch in range(num_epochs):
#         source_iter = iter(source_loader)
#         target_iter = iter(target_loader)
#         for batch_idx in range(min(len(source_loader), len(target_loader))):
#             # 取一批源域和目标域数据
#             try:
#                 x_s, y_s = next(source_iter)
#             except StopIteration:
#                 source_iter = iter(source_loader)
#                 x_s, y_s = next(source_iter)
#             try:
#                 x_t = next(target_iter)[0]
#             except StopIteration:
#                 target_iter = iter(target_loader)
#                 x_t = next(target_iter)[0]

#             x_s, y_s = x_s.to(cfg.device), y_s.to(cfg.device)
#             x_t = x_t.to(cfg.device)

#             # Step 1: 源域分类训练
#             model_e.train(); model_c1.train(); model_c2.train()
#             features = model_e(x_s)
#             p1, p2 = model_c1(features), model_c2(features)
#             loss = ce_loss(p1, y_s) + ce_loss(p2, y_s)
#             optimizer_e.zero_grad(); optimizer_c1.zero_grad(); optimizer_c2.zero_grad()
#             loss.backward()
#             optimizer_e.step(); optimizer_c1.step(); optimizer_c2.step()

#             # Step 2: 最大化分类器差异
#             model_e.eval(); model_c1.train(); model_c2.train()
#             features_s = model_e(x_s)
#             p1_s, p2_s = model_c1(features_s), model_c2(features_s)
#             with torch.no_grad():
#                 features_t = model_e(x_t)
#             p1_t, p2_t = model_c1(features_t), model_c2(features_t)
#             loss = -loss_mgr.discrepancy(p1_t, p2_t) + ce_loss(p1_s, y_s) + ce_loss(p2_s, y_s)
#             optimizer_c1.zero_grad(); optimizer_c2.zero_grad()
#             loss.backward()
#             optimizer_c1.step(); optimizer_c2.step()

#             # Step 3: 最小化分类器差异（num_k次）
#             model_e.train(); model_c1.eval(); model_c2.eval()
#             for _ in range(num_k):
#                 features_t = model_e(x_t)
#                 p1_t, p2_t = model_c1(features_t), model_c2(features_t)
#                 loss = loss_mgr.discrepancy(p1_t, p2_t)
#                 optimizer_e.zero_grad()
#                 loss.backward()
#                 optimizer_e.step()

#             # # Step 4: 领域对抗训练（可选）
#             # model_e.train(); model_d.train()
#             # source_features = model_e(x_s)
#             # target_features = model_e(x_t)
#             # features = torch.cat([source_features, target_features], dim=0)
#             # domain_labels = torch.cat([
#             #     torch.zeros(source_features.size(0), 1, device=cfg.device),
#             #     torch.ones(target_features.size(0), 1, device=cfg.device)], dim=0)
#             # features_GRL = grad_reverse(features, lam=lambda_GRL)
#             # domain_preds = model_d(features_GRL)
#             # loss = bce_loss(domain_preds, domain_labels)
#             # optimizer_d.zero_grad(); optimizer_e.zero_grad()
#             # loss.backward()
#             # optimizer_d.step(); optimizer_e.step()

#         # msg = (f"[E{epoch:03d}|B{batch_idx:04d}] "
#         #     f"CLS: {loss_cls.item():.4f}  "
#         #     f"DIS_max: {loss_dis_max.item():.4f}  "
#         #     f"DIS_min: {loss_dis_min.item():.4f}  "
#         #     f"DOM: {loss_domain.item():.4f}")
#         # logger.log(msg)            # 如果你有 logger


# def mcd_train(
#     source_loader: DataLoader,
#     target_loader: DataLoader,
#     model_e: nn.Module,
#     model_c1: nn.Module,
#     model_c2: nn.Module,
#     model_d: Optional[nn.Module],
#     optimizer_e: torch.optim.Optimizer,
#     optimizer_c1: torch.optim.Optimizer,
#     optimizer_c2: torch.optim.Optimizer,
#     optimizer_d: Optional[torch.optim.Optimizer],
#     cfg: Config,
#     logger: Optional[object] = None,
#     loss_mgr: Optional[object] = None,
#     lambda_GRL: float = 1.0,
#     num_epochs: int = 20,
#     num_k: int = 4,              # Step‑3 循环次数, MCD 论文推荐 4
#     print_freq: int = 100,       # 每 print_freq 个 batch 打印一次当前损失
#     use_domain: bool = False):    # 是否启用 Step‑4 域对抗
#     """
#     MCD 训练主循环 (以 *源域* 批次数为基准)。

#     现在外层循环固定 `len(source_loader)`，确保一个 epoch
#     会完整遍历源域数据；目标域数据不足时自动环绕。

#     其余逻辑与之前版本一致，仍提供 batch/epoch 两级损失打印。
#     """

#     if loss_mgr is None:
#         loss_mgr = LossManager()
#     if logger is None:
#         logger = Logger(log_dir="./logs", level="INFO")

#     # ce_loss = nn.CrossEntropyLoss()
#     bce_loss = nn.BCEWithLogitsLoss()

#     for epoch in range(num_epochs):
#         source_iter = iter(source_loader)
#         target_iter = iter(target_loader)

#         # epoch‑level accumulators
#         cls_loss_epoch = dis_max_epoch = dis_min_epoch = dom_loss_epoch = 0.0
#         n_batches = len(source_loader)
#         for batch_idx in range(len(source_loader)):
#             # -------------------------------------------------------------
#             # 1) 取源域 batch (必定成功，因按源域长度迭代)
#             # -------------------------------------------------------------
#             try:
#                 source_features, source_labels = next(source_iter)
#             except StopIteration:   # 安全起见，极端情况下重置
#                 source_iter = iter(source_loader)
#                 source_features, source_labels = next(source_iter)

#             # -------------------------------------------------------------
#             # 2) 取目标域 batch；若目标域耗尽则重新开始
#             # -------------------------------------------------------------
#             try:
#                 target_features = next(target_iter)[0]
#             except StopIteration:
#                 target_iter = iter(target_loader)
#                 target_features = next(target_iter)[0]

#             source_features, source_labels = source_features.to(cfg.device), source_labels.to(cfg.device)
#             target_features = target_features.to(cfg.device)

#             # ============================================================
#             # Step‑1: 源域分类
#             # ============================================================
#             model_e.train()
#             model_c1.train()
#             model_c2.train()
#             features_extract_by_e = model_e(source_features)
#             p1_source, p2_source = model_c1(features_extract_by_e), model_c2(features_extract_by_e)
#             # loss_cls = ce_loss(p1_source, source_labels) + ce_loss(p2_source, source_labels)
#             loss_cls = loss_mgr.double_classifier_loss_ce(p1_source, p2_source, source_labels)

#             optimizer_e.zero_grad()
#             optimizer_c1.zero_grad()
#             optimizer_c2.zero_grad()
#             loss_cls.backward()
#             optimizer_e.step()
#             optimizer_c1.step()
#             optimizer_c2.step()

#             # ============================================================
#             # Step‑2: 最大化分类器差异
#             # ============================================================
#             # model_e.eval()
#             # model_c1.train()
#             # model_c2.train()
#             # # 使用 detach 的源域特征避免更新特征提取器
#             # with torch.no_grad():
#             #     features_extract_by_e_detached = model_e(source_features)
#             # p1_source_discrepancy = model_c1(features_extract_by_e_detached)
#             # p2_source_discrepancy = model_c2(features_extract_by_e_detached)

#             # with torch.no_grad():
#             #     f_t = model_e(target_features)
#             # p1_t = model_c1(f_t)
#             # p2_t = model_c2(f_t)
#             # disc_val = loss_mgr.discrepancy(p1_t, p2_t)
#             # loss_dis_max = -disc_val + ce_loss(p1_source_discrepancy, source_labels) + ce_loss(p2_source_discrepancy, source_labels)

#             # optimizer_c1.zero_grad()
#             # optimizer_c2.zero_grad()
#             # loss_dis_max.backward()
#             # optimizer_c1.step()
#             # optimizer_c2.step()
#             # ================ 以上是备份 ================

#             model_e.eval()
#             model_c1.train()
#             model_c2.train()
#             # 将 step 2 修改为纯粹的对抗，避免因损失函数过于复杂而导致训练不稳定
#             with torch.no_grad():
#                 f_t = model_e(target_features)
#             p1_t = model_c1(f_t)
#             p2_t = model_c2(f_t)
#             loss_dis_max = -loss_mgr.discrepancy(p1_t, p2_t)

#             optimizer_c1.zero_grad()
#             optimizer_c2.zero_grad()
#             loss_dis_max.backward()
#             optimizer_c1.step()
#             optimizer_c2.step()

#             # ============================================================
#             # Step‑3: 最小化分类器差异
#             # ============================================================
#             # model_e.train()
#             # model_c1.eval()
#             # model_c2.eval()
#             # for _ in range(num_k):
#             #     f_t = model_e(target_features)
#             #     p1_t, p2_t = model_c1(f_t), model_c2(f_t)
#             #     loss_dis_min = loss_mgr.discrepancy(p1_t, p2_t)
#             #     optimizer_e.zero_grad()
#             #     loss_dis_min.backward()
#             #     optimizer_e.step()

#             model_e.train()
#             model_c1.eval()
#             model_c2.eval()
#             f_t = model_e(target_features)
#             p1_t, p2_t = model_c1(f_t), model_c2(f_t)
#             loss_dis_min = loss_mgr.discrepancy(p1_t, p2_t)
#             optimizer_e.zero_grad()
#             loss_dis_min.backward()
#             optimizer_e.step()

#             # ============================================================
#             # Step‑4: 域对抗 (可选)
#             # ============================================================
#             if use_domain and model_d is not None and optimizer_d is not None:
#                 model_e.train(); model_d.train()
#                 features_extract_by_e = model_e(source_features)
#                 f_t = model_e(target_features)
#                 features = torch.cat([features_extract_by_e, f_t], dim=0)
#                 domain_labels = torch.cat([
#                     torch.zeros(features_extract_by_e.size(0), 1, device=cfg.device),
#                     torch.ones(f_t.size(0), 1, device=cfg.device)
#                 ], dim=0)
#                 features_grl = grad_reverse(features, lam=lambda_GRL)
#                 domain_preds = model_d(features_grl)
#                 loss_domain = bce_loss(domain_preds, domain_labels)
#                 optimizer_d.zero_grad(); optimizer_e.zero_grad()
#                 loss_domain.backward()
#                 optimizer_d.step(); optimizer_e.step()
#             else:
#                 loss_domain = torch.tensor(0.0, device=cfg.device)

#             # ----------------- 统计 & 打印 -----------------
#             cls_loss_epoch += loss_cls.item()
#             dis_max_epoch += (-loss_dis_max).item()
#             dis_min_epoch += loss_dis_min.item()
#             dom_loss_epoch += loss_domain.item()

#             if (batch_idx + 1) % print_freq == 0:
#                 msg = (f"[E{epoch:02d}|B{batch_idx + 1:03d}] "
#                        f"Classification loss: {loss_cls.item():.4f}  "
#                        f"Discrepancy maximization: {(-loss_dis_max).item():.4f}  "
#                        f"Discrepancy minimization: {loss_dis_min.item():.4f}  "
#                        f"Domain adversarial loss: {loss_domain.item():.4f}")
#                 logger.log(msg)

#         # model_e.eval()
#         # all_features = []
#         # all_labels = []
#         # all_domains = []

#         # with torch.no_grad():
#         #     for xs, ys in source_loader:
#         #         xs, ys = xs.to(cfg.device), ys.to(cfg.device)
#         #         fs = model_e(xs)
#         #         all_features.append(fs)
#         #         all_labels.append(ys)
#         #         all_domains.append(torch.zeros_like(ys))
            
#         #     for xt in target_loader:
#         #         xt = xt[0].to(cfg.device)
#         #         ft = model_e(xt)
#         #         dummy_labels = torch.full((xt.size(0),), -1, device=cfg.device)
#         #         all_features.append(ft)
#         #         all_labels.append(dummy_labels)
#         #         all_domains.append(torch.ones(xt.size(0), device=cfg.device))
#         # features = torch.cat(all_features, dim=0).cpu().numpy()
#         # # labels = all_labels.cpu().numpy()
#         # domains = torch.cat(all_domains, dim=0).cpu().numpy()

#         # reducer = TSNE(n_components=2, init='pca', random_state=42)
#         # features_2d = reducer.fit_transform(features)

#         # plt.figure(figsize=(8, 6))
#         # for domain_id, marker, name in zip([0, 1], ['o', '^'], ['source', 'target']):
#         #     idx = domains == domain_id
#         #     plt.scatter(features_2d[idx, 0], 
#         #                 features_2d[idx, 1], 
#         #                 alpha=0.5,
#         #                 marker=marker,
#         #                 label=name)
#         # plt.legend()
#         # plt.title(f"Feature distribution at epoch: {epoch + 1}")
#         # plt.show()

#         # ----------------- epoch 级汇总 -----------------
#         avg_cls = cls_loss_epoch / n_batches
#         avg_dmax = dis_max_epoch / n_batches
#         avg_dmin = dis_min_epoch / n_batches
#         avg_dom = dom_loss_epoch / n_batches if use_domain else 0.0
#         epoch_msg = (f"[E{epoch:03d} Summary] "
#                      f"Classification loss: {avg_cls:.4f}  Discrepancy maximization: {avg_dmax:.4f}  "
#                      f"Discrepancy minimization: {avg_dmin:.4f}  Domain adversarial loss: {avg_dom:.4f}")
#         logger.log(epoch_msg)

def evaluate_accuracy(data_loader: DataLoader,
                      model_e: nn.Module,
                      model_c1: nn.Module,
                      model_c2: nn.Module,
                      cfg: Config,
                      logger: Logger,
                      domain_name: str):
    model_e.eval()
    model_c1.eval()
    model_c2.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            features = model_e(x)
            pred = (model_c1(features) + model_c2(features)) / 2
            y_pred = pred.argmax(dim=1).cpu().numpy()
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(y_pred)
    accuracy = balanced_accuracy_score(all_y_true, all_y_pred)
    logger.log("== Accuracy in {}: {:.2f}% ==", domain_name, accuracy * 100, level="INFO")
    model_e.train()    
    model_c1.train()
    model_c2.train()
    return accuracy


def mcd_train(
    source_loader: DataLoader,
    target_loader: DataLoader,
    model_e: nn.Module,
    model_c1: nn.Module,
    model_c2: nn.Module,
    model_d: Optional[nn.Module],
    optimizer_e: torch.optim.Optimizer,
    optimizer_c1: torch.optim.Optimizer,
    optimizer_c2: torch.optim.Optimizer,
    optimizer_d: Optional[torch.optim.Optimizer],
    cfg: Config,
    logger: Optional[object] = None,
    loss_mgr: Optional[object] = None,
    lambda_GRL: float = 1.0,
    # num_epochs: int = 60,
    # num_k: int = 4,              # Step‑3 循环次数, MCD 论文推荐 4
    print_freq: int = 100,       # 每 print_freq 个 batch 打印一次当前损失
    use_domain: bool = False):    # 是否启用 Step‑4 域对抗
    """
    MCD 训练主循环 (以 *源域* 批次数为基准)。

    现在外层循环固定 `len(source_loader)`，确保一个 epoch
    会完整遍历源域数据；目标域数据不足时自动环绕。

    其余逻辑与之前版本一致，仍提供 batch/epoch 两级损失打印。
    """

    if loss_mgr is None:
        loss_mgr = LossManager()
    if logger is None:
        logger = Logger(log_dir="./logs", level="INFO")

    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.max_epochs):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        # epoch‑level accumulators
        cls_loss_epoch = dis_max_epoch = dis_min_epoch = dom_loss_epoch = dom_loss_epoch_d = dom_loss_epoch_e = 0.0
        n_batches = len(source_loader)
        for batch_idx in range(len(source_loader)):
            # -------------------------------------------------------------
            # 1) 取源域 batch (必定成功，因按源域长度迭代)
            # -------------------------------------------------------------
            try:
                source_features, source_labels = next(source_iter)
            except StopIteration:   # 安全起见，极端情况下重置
                source_iter = iter(source_loader)
                source_features, source_labels = next(source_iter)

            # -------------------------------------------------------------
            # 2) 取目标域 batch；若目标域耗尽则重新开始
            # -------------------------------------------------------------
            try:
                target_features = next(target_iter)[0]
            except StopIteration:
                target_iter = iter(target_loader)
                target_features = next(target_iter)[0]

            source_features, source_labels = source_features.to(cfg.device), source_labels.to(cfg.device)
            target_features = target_features.to(cfg.device)

            # ============================================================
            # Step‑1: 源域分类
            # ============================================================
            model_e.train()
            model_c1.train()
            model_c2.train()
            fs = model_e(source_features)
            p1_source, p2_source = model_c1(fs), model_c2(fs)
            loss_cls = loss_mgr.double_classifier_loss_ce(p1_source, p2_source, source_labels)

            optimizer_e.zero_grad()
            optimizer_c1.zero_grad()
            optimizer_c2.zero_grad()
            loss_cls.backward()
            optimizer_e.step()
            optimizer_c1.step()
            optimizer_c2.step()

            # ============================================================
            # Step‑2: 最大化分类器差异
            # ============================================================
            model_e.eval()
            model_c1.train()
            model_c2.train()
            # 将 step 2 修改为纯粹的对抗，避免因损失函数过于复杂而导致训练不稳定
            # with torch.no_grad():
            #     f_t = model_e(target_features)
            ft = model_e(target_features)
            p1_t = model_c1(ft)
            p2_t = model_c2(ft)
            loss_dis_max = -loss_mgr.discrepancy_stable(p1_t, p2_t)

            optimizer_c1.zero_grad()
            optimizer_c2.zero_grad()
            loss_dis_max.backward()
            optimizer_c1.step()
            optimizer_c2.step()

            # ============================================================
            # Step‑3: 最小化分类器差异
            # ============================================================
            model_e.train()
            model_c1.eval()
            model_c2.eval()
            ft = model_e(target_features)
            p1_t, p2_t = model_c1(ft), model_c2(ft)
            loss_dis_min = loss_mgr.discrepancy_stable(p1_t, p2_t)
            # 尝试引入 mmd 损失
            fs_mmd = model_e(source_features)
            ft_mmd = model_e(target_features)
            loss_mmd = 0.05 * loss_mgr.mmd_loss(fs_mmd, ft_mmd, cfg)
            loss_dis_min += loss_mmd

            optimizer_e.zero_grad()
            loss_dis_min.backward()
            optimizer_e.step()

            # ============================================================
            # Step‑4: 域对抗 (可选)
            # ============================================================
            # if use_domain and model_d is not None and optimizer_d is not None:
            #     model_e.train(); model_d.train()
            #     features_extract_by_e = model_e(source_features)
            #     f_t = model_e(target_features)
            #     features = torch.cat([features_extract_by_e, f_t], dim=0)
            #     domain_labels = torch.cat([
            #         torch.zeros(features_extract_by_e.size(0), 1, device=cfg.device),
            #         torch.ones(f_t.size(0), 1, device=cfg.device)
            #     ], dim=0)
            #     features_grl = grad_reverse(features, lam=lambda_GRL)
            #     domain_preds = model_d(features_grl)
            #     loss_domain = bce_loss(domain_preds, domain_labels)
            #     optimizer_d.zero_grad(); optimizer_e.zero_grad()
            #     loss_domain.backward()
            #     optimizer_d.step(); optimizer_e.step()
            # else:
            #     loss_domain = torch.tensor(0.0, device=cfg.device)
            # ====================== 以上为备份 ======================
            # update domain_discriminator
            model_e.eval()
            model_d.train()
            fs1 = model_e(source_features)
            ft1 = model_e(target_features)
            features = torch.cat([fs1, ft1], dim=0)
            domain_labels = torch.cat([
                torch.zeros(fs1.size(0), dtype=torch.long, device=cfg.device),
                torch.ones(ft1.size(0), dtype=torch.long, device=cfg.device)
            ], dim=0)
            # domain_labels = torch.cat([
            #     torch.zeros(fs1.size(0), 1, device=cfg.device),
            #     torch.ones(ft1.size(0), 1, device=cfg.device)
            # ], dim=0)
            domain_preds = model_d(features.detach())
            # loss_d = bce_loss(domain_preds, domain_labels)
            loss_d = loss_mgr.domain_loss(domain_preds, domain_labels)

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            # update features extractor
            model_e.eval()
            model_d.eval()
            fs2 = model_e(source_features)
            ft2 = model_e(target_features)
            domain_labels2 = torch.cat([
                torch.zeros(fs2.size(0), dtype=torch.long, device=cfg.device),
                torch.ones(ft2.size(0), dtype=torch.long, device=cfg.device)
            ], dim=0)
            features = torch.cat([fs2, ft2], dim=0)
            features_grl = grad_reverse(features, lambda_GRL)
            domain_preds2 = model_d(features_grl)
            # loss_e_adv = bce_loss(domain_preds, domain_labels)
            loss_e_adv = loss_mgr.domain_loss(domain_preds2, domain_labels2)

            # optimizer_e.zero_grad()
            # loss_e_adv.backward()
            # optimizer_e.step()
            loss_domain = loss_d + loss_e_adv

            # ----------------- 统计 & 打印 -----------------
            cls_loss_epoch += loss_cls.item()
            dis_max_epoch += (-loss_dis_max).item()
            dis_min_epoch += loss_dis_min.item()
            dom_loss_epoch += loss_domain.item()
            dom_loss_epoch_d += loss_d.item()
            dom_loss_epoch_e += loss_e_adv.item()

            if (batch_idx + 1) % print_freq == 0:
                msg = (f"[E{epoch:02d}|B{batch_idx + 1:03d}]:\n"
                       f"Classification loss: {loss_cls.item():.4f}\n"
                       f"Discrepancy maximization: {(-loss_dis_max).item():.4f}\n"
                       f"Discrepancy minimization: {loss_dis_min.item():.4f}\n"
                       f"Domain adversarial loss: {loss_domain.item():.4f}\n"
                       f"Discriminator loss: {loss_d.item():.4f}\n"
                       f"Feature extractor loss: {loss_e_adv.item():.4f}")
                logger.log(msg)
                with torch.no_grad():
                    domain_probs = torch.softmax(domain_preds, dim=1)
                    pred_labels = domain_probs.argmax(dim=1)
                    acc = (pred_labels == domain_labels).float().mean().item()
                    logger.log("Domain discriminator accuracy: {:.2f}%", acc * 100)

        # ----------------- epoch 级汇总 -----------------
        avg_cls = cls_loss_epoch / n_batches
        avg_dmax = dis_max_epoch / n_batches
        avg_dmin = dis_min_epoch / n_batches
        avg_dom = dom_loss_epoch / n_batches if use_domain else 0.0
        avg_dom_d = dom_loss_epoch_d / n_batches
        avg_dom_e = dom_loss_epoch_e / n_batches
        epoch_msg = (f"\n========== [E{epoch:03d} Summary] ==========\n"
                     f"Classification loss: {avg_cls:.4f}\n"
                     f"Discrepancy maximization: {avg_dmax:.4f}\n"
                     f"Discrepancy minimization: {avg_dmin:.4f}\n"
                     f"Domain adversarial loss: {avg_dom:.4f}\n"
                     f"Discriminator loss: {avg_dom_d:.4f}\n"
                     f"Feature extractor loss: {avg_dom_e:.4f}\n"
                     f"========== [E{epoch:03d} Summary] ==========\n")
        logger.log(epoch_msg)
        logger.log("=======================================")
        evaluate_accuracy(source_loader, model_e, model_c1, model_c2, cfg, logger, "source_domain")
        evaluate_accuracy(target_loader, model_e, model_c1, model_c2, cfg, logger, "target_domain")
        logger.log("=======================================\n")


# ------------------------------
# 算法二：Custom Domain Adaptation
# ------------------------------
# 定义基本残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        identity = self.residual_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)
    