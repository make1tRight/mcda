import torch
import torch.nn as nn
import torch.nn.functional as F
import LossManager
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.autograd import Function
import warnings
from Logger import *

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
    category=UserWarning
)


# ------------------------------
# 模块一：特征提取器
# ------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)
    

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2))

    def forward(self, x):
        return self.encoder(x)


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.1):
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
            d_model=hidden_dim, nhead=n_heads, dropout=dropout,
            batch_first=True, norm_first=True  # 提前归一化增加稳定性
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 4. 聚合输出（比如平均池化）
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch_size, input_dim) → (batch_size, 1, hidden_dim)
        x = self.input_proj(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("After input_proj: NaN or Inf")
        x = self.input_norm(x)

        x = x.unsqueeze(1) + self.pos_embedding

        # Transformer 编码
        x = self.encoder(x)  # (batch_size, 1, hidden_dim)
        # 池化成一个向量
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, 1)
        x = self.pool(x).squeeze(-1)  # (batch_size, hidden_dim)

        return x


# ------------------------------
# 模块二：分类器
# ------------------------------
class Classifier(nn.Module):
    def __init__(self, feature_dim_1, feature_dim_2, feature_dim_3, num_classes):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim_1, feature_dim_2),
            nn.ReLU(),
            nn.Linear(feature_dim_2, feature_dim_3),
            nn.ReLU(),
            nn.Linear(feature_dim_3, num_classes)
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
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
# 模块四：MCD 训练函数
# ------------------------------
def mcd_train_3(source_x, source_y, target_x,
                model_e, model_c1, model_c2, model_d,
                optimizer_e, optimizer_c1, optimizer_c2, optimizer_d,
                step1_iter, step2_iter, step3_iter, step4_iter,
                logger=None, loss_mgr=None, lambda_GRL=1.0):
    if loss_mgr is None:
        loss_mgr = LossManager.LossManager()
    if logger is None:
        logger = Logger(log_dir="./logs", level="INFO")

    # Logger(log_dir)
    # Step 1: 用源域有标签样本训练 E+C1+C2
    model_e.train()
    model_c1.train()
    model_c2.train()

    for step in range(step1_iter):
        features = model_e(source_x)
        out1 = model_c1(features)
        out2 = model_c2(features)
        if torch.isnan(out1).any() or torch.isinf(out1).any():
            # print("out1 contains NaN or Inf")
            logger.log("out1 contains NaN or Inf")
        if torch.isnan(out2).any() or torch.isinf(out2).any():
            # print("out2 contains NaN or Inf")
            logger.log("out2 contains NaN or Inf")
        loss_s = loss_mgr.double_classifier_loss_ce(out1, out2, source_y)

        optimizer_e.zero_grad()
        optimizer_c1.zero_grad()
        optimizer_c2.zero_grad()
        loss_s.backward()
        optimizer_e.step()
        optimizer_c1.step()
        optimizer_c2.step()

        if (step + 1) % 100 == 0 or step == 0 or (step + 1) == step1_iter:
            with torch.no_grad():
                pred = (out1 + out2) / 2
                pred_list = torch.bincount(pred.argmax(dim=1), minlength=loss_mgr.num_classes)
                # print(f"[step 1 - Iter {step + 1}/{step1_iter}]")
                logger.log("[step 1 - Iter {}/{}]", step + 1, step1_iter)
                # print(f"step 1 loss: {loss_s.item(): .4f},"
                      # f" pred distribution: {pred_list.cpu().numpy()}")
                logger.log("step 1 loss: {: .4f}", loss_s.item())
                logger.log("pred distribution: {}", pred_list.cpu().numpy())

    # Step 2: 固定E，最大化目标域预测差异
    for step in range(step2_iter):
        features_t = model_e(target_x).detach()
        out1 = model_c1(features_t)
        out2 = model_c2(features_t)
        loss_diff = -loss_mgr.discrepancy_loss(out1, out2)

        optimizer_c1.zero_grad()
        optimizer_c2.zero_grad()
        loss_diff.backward()
        optimizer_c1.step()
        optimizer_c2.step()

        if (step + 1) % 100 == 0 or step == 0 or (step + 1) == step2_iter:
            # 打印 loss 和预测分布
            with torch.no_grad():
                pred1 = out1.argmax(dim=1)
                pred2 = out2.argmax(dim=1)
                pred1_dist = torch.bincount(pred1, minlength=loss_mgr.num_classes)
                pred2_dist = torch.bincount(pred2, minlength=loss_mgr.num_classes)

                logger.log("[step 2 - Iter {}/{}]", step + 1, step2_iter)
                logger.log("loss: {: .4f}", loss_diff.item())
                logger.log("C1 pred distribution: {}", pred1_dist.cpu().numpy())
                logger.log("C2 pred distribution: {}", pred2_dist.cpu().numpy())

    # Step 3: 固定C1/C2，更新E最小化预测差异
    for step in range(step3_iter):
        features_t = model_e(target_x)
        out1 = model_c1(features_t)
        out2 = model_c2(features_t)
        loss_diff = loss_mgr.discrepancy_loss(out1, out2)

        optimizer_e.zero_grad()
        loss_diff.backward()
        optimizer_e.step()

        if (step + 1) % 100 == 0 or step == 0 or (step + 1) == step3_iter:
            # 打印 loss 和预测分布
            with torch.no_grad():
                pred = (out1 + out2) / 2
                pred_labels = pred.argmax(dim=1)
                pred_dist = torch.bincount(pred_labels, minlength=loss_mgr.num_classes)

                # print(f"[step 3 - Iter {step + 1}/{step3_iter}]")
                # print(f"loss: {loss_diff.item(): .4f},")
                # print(f"pred distribution: {pred_dist.cpu().numpy()}")
                logger.log("[step 3 - Iter {}/{}]", step + 1, step3_iter)
                logger.log("loss: {: .4f}", loss_diff.item())
                logger.log("pred distribution: {}", pred_dist.cpu().numpy())

    # Step 4: 领域对抗训练
    for step in range(step4_iter):
        model_d.train()
        model_e.train()

        features_s = model_e(source_x)
        features_t = model_e(target_x)
        features = torch.cat([features_s, features_t], dim=0)

        domain_labels = torch.cat([
            torch.zeros(features_s.size(0), 1),
            torch.ones(features_t.size(0), 1)
        ], dim=0).to(features.device)

        features_GRL = grad_reverse(features, lam=lambda_GRL)
        domain_preds = model_d(features_GRL)

        loss_d = F.binary_cross_entropy(domain_preds, domain_labels)

        optimizer_d.zero_grad()
        optimizer_e.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        optimizer_e.step()

        if (step + 1) % 10 == 0 or step == 0 or (step + 1) == step4_iter:
            logger.log("[step 4 - Iter {}/{}]", step + 1, step4_iter)
            logger.log("loss: {: .4f}", loss_d.item())


# ------------------------------
# 算法二：Custom Domain Adaptation
# ------------------------------
# 定义基本残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        identity = self.residual_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# RecResNet backbone
class RecResNetBackbone(nn.Module):
    def __init__(self, input_channels=1, features_dim=128):
        super(RecResNetBackbone, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 残差模块结构，block可调整
        self.layer2 = ResidualBlock(64, 128, downsample=True)
        self.layer3 = ResidualBlock(128, 256, downsample=True)
        self.layer4 = ResidualBlock(256, 512, downsample=True)

        # 全局平均池化 + 特征降维
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, features_dim)

    def forward(self, x):
        x = self.layer1(x)  # 初始卷积 + BN + ReLU + 池化
        x = self.layer2(x)  # 残差块1
        x = self.layer3(x)  # 残差块2
        x = self.layer4(x)  # 残差块3
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.fc(x)   # 输出特征维度为feature_dim


# CustomDomainAdaptation框架
class CdaModel(nn.Module):
    def __init__(self, features_dim=128, num_classes=4):
        super(CdaModel, self).__init__()
        self.GSource = RecResNetBackbone(features_dim=features_dim)  # 源域网络
        self.GTarget = RecResNetBackbone(features_dim=features_dim)  # 目标域网络
        self.classifier = nn.Linear(features_dim, num_classes)

    def forward(self, x_source, x_target):
        feature_source = self.GSource(x_source)
        feature_target = self.GTarget(x_target)
        out_source = self.classifier(feature_source)
        out_target = self.classifier(feature_target)
        return out_source, out_target, feature_source, feature_target
