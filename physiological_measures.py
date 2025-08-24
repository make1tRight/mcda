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
from logger import Logger
from typing import Optional, Dict, List



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
                 low: int,
                 mid: int,
                 high: int,
                 binary_threshold: int,
                 num_classes: int):
        """
        
        """
        self.num_classes = num_classes
        self.low_start = low
        self.mid_start = mid
        self.high_start = high
        self.binary_threshold = binary_threshold
        
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

class MultimodalLoader:
  def __init__(self, cfg: Config, logger: Logger, lbl_classifier: LabelClassifier):
    self.cfg = cfg
    self.logger = logger
    self.lbl_classifier = lbl_classifier
    
    # 计算期望的总特征维度
    self.expected_feature_num = sum(m.feature_num for m in self.cfg.modals)
    self.modal_schema = {m.type: [f"{m.type}_f{i}" for i in range(m.feature_num)]
                     for m in self.cfg.modals}

  def LoadMultimodalData(self) -> pd.DataFrame:
    """
    加载多模态数据，处理缺失模态
    """
    all_data = []
    for subject in self.cfg.subjects:
      data = self._loadSubjectData(subject)
      if data is not None:
        all_data.append(data)
    
    if not all_data:
        raise ValueError("cannot load any subject data")
    return pd.concat(all_data, ignore_index=True)

  def _loadSubjectData(self, subject: str) -> Optional[pd.DataFrame]:
    """
    加载单个被试的数据
    """
    subject_data: Dict[str, pd.DataFrame] = {}
    available_modal_types: List[str] = []

    for modal_config in self.cfg.modals:
      modal_type = modal_config.type
      file_path = f'{self.cfg.data_path}/{subject}/20width-4step/combined_{modal_type}_features.csv'
      try:
        data = pd.read_csv(file_path, na_values=['--', '-', 'NA', 'NaN', 'nan', ''])
        data.replace(['--', '-', 'NA', 'NaN', 'nan', ''], np.nan, inplace=True)
        if 'relative_time' not in data.columns or 'MWL_Rating' not in data.columns:
          self.logger.log("No relative_time or MWL_Rating column in [subject: {}] data", 
                          subject, level="WARNING")
          continue
        subject_data[modal_type] = data
        available_modal_types.append(modal_type)
      except FileNotFoundError:
        self.logger.log("Missing file for subject: {} modal type: {}", subject, modal_type)
        continue
    
    if not available_modal_types:
      self.logger.log("No available modal type for subject: {}", subject)
      return None
    
    combined_data = self._combineModalities(subject_data, available_modal_types, subject)
    self.logger.log("Merged subject {}: all features shape={}", subject, combined_data.shape, level="INFO")
    return combined_data

  def _build_modal_block(self, 
                         modal_type: str,
                         df: Optional[pd.DataFrame],
                         base_times: List[float],
                         feature_num: int,
                         default_value: float) -> pd.DataFrame:
    """
    构建模态块
    """
    cols = self.modal_schema[modal_type]
    if df is None:
      return pd.DataFrame({
        'relative_time': base_times,
        **{col: default_value for col in cols}})
    
    feature_cols = [c for c in df.columns if c not in ['relative_time', 'MWL_Rating']]
    feature_cols = sorted(feature_cols)
    k = len(feature_cols)

    if k >= feature_num:
       use_cols = feature_cols[:feature_num]
       block = df[['relative_time'] + use_cols].copy()
       rename_map = {old: new for old, new in zip(use_cols, cols[:feature_num])}
       block = block.rename(columns=rename_map)
    else:
      use_cols = feature_cols
      block = df[['relative_time'] + use_cols].copy()
      rename_map = {old: new for old, new in zip(use_cols, cols[:k])}
      block = block.rename(columns=rename_map)
      for miss_col in cols[k:]:
        block[miss_col] = default_value
        
    return block[['relative_time'] + cols]

  def _combineModalities(self, 
                         subject_data: Dict, 
                         avail_model_type: List[str], 
                         subject: str) -> pd.DataFrame:
    """
    合并多模态数据
    """
    # [收集时间戳]
    tol = 3.0
    all_timestamp = sorted({t for df in subject_data.values() for t in df['relative_time'].dropna()})
    base_times = []
    if all_timestamp:
      rep = all_timestamp[0]
      base_times.append(rep)
      for t in all_timestamp[1:]:
        if t - rep > tol:
          rep = t
          base_times.append(rep)
    total_features = pd.DataFrame({'relative_time': base_times})

    if not all_timestamp:
      raise ValueError("No timestamp found in [subject: {}] data", subject)
    
    # [合并特征]
    features = []
    for m in self.cfg.modals:
      modal_type = m.type
      if modal_type in avail_model_type:
        block = self._build_modal_block(modal_type,
                                        subject_data[modal_type],
                                        base_times,
                                        m.feature_num,
                                        m.default_value)
      else:
        block = self._build_modal_block(modal_type,
                                        None,
                                        base_times,
                                        m.feature_num,
                                        m.default_value)
      features.append(block)

    for block in features:
      # total_features = total_features.merge(block, on='relative_time', how='left')
      total_features = pd.merge_asof(total_features,
                                     block, 
                                     on='relative_time', 
                                     direction='backward', 
                                     tolerance=tol)
    

    # [合并标签数据]
    labels = []
    for modal_config in self.cfg.modals:
      modal_type = modal_config.type
      if modal_type in avail_model_type and \
        modal_type in subject_data and \
        'MWL_Rating' in subject_data[modal_type].columns:
          labels.append(
            subject_data[modal_type][['relative_time', 'MWL_Rating']]
            .rename(columns={'MWL_Rating': f'MWL_Rating__{modal_type}'}))
      
    if labels:
      lbl = labels[0]
      for extra in labels[1:]:
        lbl = lbl.merge(extra, on='relative_time', how='outer')
      label_cols = [c for c in lbl.columns if c.startswith('MWL_Rating__')]
      lbl['MWL_Rating'] = lbl[label_cols].bfill(axis=1).ffill(axis=1).iloc[:, 0]
      lbl = lbl[['relative_time', 'MWL_Rating']]
      total_features = total_features.merge(lbl, on='relative_time', how='left')
      total_features['MWL_Rating'] = total_features['MWL_Rating'].ffill().bfill()
    else:
      total_features['MWL_Rating'] = np.nan

    # 处理缺失值
    features_only = [c for c in total_features.columns if c not in ['relative_time', 'MWL_Rating']]
    if len(features_only) != self.expected_feature_num:
      self.logger.log("Feature dimension mismatch for subject: {}, expected: {}, got: {}",
                      subject, self.expected_feature_num, len(features_only), level="WARNING")
      raise ValueError(f"Subject: {subject}: expected: {self.expected_feature_num}, got: {len(features_only)}")
    
    total_features[features_only] = total_features[features_only].fillna(0)

    # 标签分类
    total_features['MWL_Rating'] = total_features['MWL_Rating'].apply(self.lbl_classifier.classify)

    # total_features['subject_id'] = subject
    total_features = total_features.assign(subject_id=subject)
    total_features = total_features.sort_values(by='relative_time').reset_index(drop=True)
    total_features = total_features.drop_duplicates(subset=features_only, keep='first')
    self.logger.log("Success to combine modalities for subject: {}, shape: {}", 
                    subject, total_features.shape, level="INFO")
    return total_features
