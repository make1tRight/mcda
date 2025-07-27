import torch
from torch.utils.data import Dataset
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay)
from config import Config


# ------------------------------
# 个体归一化
# ------------------------------
def per_feature_individual_zscore(features, exclude_cols=['MWL_Rating']):
    features_norm = features.copy()
    features_cols = [col for col in features.columns if col not in exclude_cols]
    for col in features_cols:
        mean = features[col].mean()
        std = features[col].std() + 1e-6
        features_norm[col] = (features[col] - mean) / std
    return features_norm


# ------------------------------
# 静息归一化
# ------------------------------
def normalize_by_rest_state(df: pd.DataFrame,
                            rest_duration_minutes: int,
                            sampling_rate: int,
                            label_column: str = 'MWL_Rating') -> pd.DataFrame:
    """
    基于实验开始前静息阶段进行 Z-score 标准化。

    参数：
    - df: DataFrame，包含按时间顺序排列的模态特征与标签
    - rest_duration_minutes: 静息阶段时长（分钟）
    - sampling_rate: 数据采样频率（每秒多少行）
    - label_column: 标签列名，默认是 'MWL_Rating'

    返回：
    - 标准化后的 DataFrame（标签列保持不变）
    """
    # 静息阶段数据行数
    rest_rows = rest_duration_minutes * 60 * sampling_rate
    # 分离标签
    features = df.drop(columns=label_column)
    labels = df[label_column]

    # 计算静息状态下每个特征的均值和标准差
    rest_means = features.iloc[:rest_rows].mean()
    rest_stds = features.iloc[:rest_rows].std()

    # 防止除以0
    rest_stds[rest_stds == 0] = 1e-8

    # Z-score 标准化
    normalized_features = (features - rest_means) / rest_stds
    # 合并标签列
    normalized_df = pd.concat([normalized_features, labels], axis=1)

    return normalized_df


class LabelClassifier:
    def __init__(self, 
                 cfg: Config):
        """
        :param low_start: basic 模式下低负荷的起始分数
        :param mid_start: basic 模式下中负荷的起始分数
        :param high_start: basic 模式下高负荷的起始分数
        """
        self.num_classes = cfg.num_classes
        self.low_start = cfg.low_level
        self.mid_start = cfg.mid_level
        self.high_start = cfg.high_level
        self.binary_threshold = cfg.binary_threshold
        
    def classify(self, rating):
        """
        将单个标签值分类为 0/1/2。
        :param x: 单个 MWL_Rating 值
        :return: 类别标签 0/1/2
        """
        if self.num_classes == 3:
            if rating < self.mid_start:
                return 0
            elif self.mid_start <= rating < self.high_start:
                return 1
            else:
                return 2
        elif self.num_classes == 2:
            if rating <= self.binary_threshold:
                return 0
            else:
                return 1

def load_eeg_data(cfg: Config):
    """
    加载多个被试的 EEG 数据，并统一处理标签和添加被试编号列。

    :param subjects: 被试编号列表
    :param base_path: 基础文件路径，包含所有被试的子文件夹
    :param low_start: 低类别最低分数
    :param mid_start: 中类别最低分数
    :param high_start: 高类别最低分数
    :return: 合并后的 DataFrame
    """
    all_data = []
    for subject in cfg.subjects:
        file_path = f'{cfg.data_path}/{subject}/20width-4step/combined_eeg_features.csv'
        df = pd.read_csv(file_path)
        # 特征归一化
        normalized_df = normalize_by_rest_state(df, 
                                                rest_duration_minutes=5, 
                                                sampling_rate=256)
        # 标签分界类
        classifier = LabelClassifier(cfg)
        # 统一标签处理
        normalized_df['MWL_Rating'] = \
            normalized_df['MWL_Rating'].apply(classifier.classify)
        # 添加被试编号列
        normalized_df['subject_id'] = subject

        all_data.append(normalized_df)
    # 合并所有数据并返回
    return pd.concat(all_data, ignore_index=True)


def impute_missing_values_by_knn(x, n_neighbors=5):
    """
    使用 KNN 算法填充缺失值。

    :param x: 特征数据
    :param n_neighbors: 用于 KNN 填充的邻居数量，默认为 5
    :return: 填充后的特征数据
    """
    knn_imp = KNNImputer(n_neighbors=n_neighbors)
    x_imp = knn_imp.fit_transform(x)
    return x_imp


def balance_data(x_train, y_train, random_state=42):
    """
    使用 SMOTE 对训练集进行重采样，平衡类别。

    Parameters:
    - X_train: ndarray，训练集特征
    - y_train: pandas.Series，训练集标签
    - random_state: int，随机种子

    Returns:
    - X_train_resampled, y_train_resampled: 重采样后的训练集特征和标签
    """
    smote = SMOTE(random_state=random_state)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    counter_before = Counter(y_train)
    formatted_before = ' '.join([f'{k}: {v}' for k, v in sorted(counter_before.items())])
    counter_after = Counter(y_train_resampled)
    formatted_after = ' '.join([f'{k}: {v}' for k, v in sorted(counter_after.items())])
    print(f"before balance: {formatted_before}")
    print(f"after balanced: {formatted_after}")
    return x_train_resampled, y_train_resampled


# ------------------------------
# 数据集类
# ------------------------------
class EEGECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def plot_confusion_matrix(y_true, y_pred, subject_id, save_path=None, cmap=plt.cm.Blues, show=False):
    """
    计算并显示混淆矩阵。

    Parameters:
    - y_true: list 或 np.ndarray，真实标签
    - y_pred: list 或 np.ndarray，预测标签
    - cmap: matplotlib colormap，色图，默认蓝色调色板

    Returns:
    - None，直接显示混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=cmap)  # 使用指定的色图
    plt.title(f"Confusion Matrix - Subject: {subject_id}")  # 显示被试编号
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    if show:
        plt.show()


# class PairedData(object):
#     def __init__(self, dataloader1, dataloader2, max_datasets_size=float("inf")):
      