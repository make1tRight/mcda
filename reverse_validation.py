from config import Config
import pandas as pd
import domain_adaptation as da
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from loss_manager import LossManager
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import (
    balanced_accuracy_score, 
    f1_score, 
    precision_score,
    recall_score, 
    roc_auc_score)
from sklearn.preprocessing import label_binarize
import numpy as np
import copy
import physiological_measures as phm
from logger import Logger

class ReverseValidateModelTrainer:
  """
  源域完整训练 + 目标域1次测试
  """

  def __init__(self, cfg: Config, logger=None):
    self.cfg = cfg
    self.logger = logger

  def fit_predict(self, source_data: pd.DataFrame, target_data: pd.DataFrame):
    """
    Parameters
    ----------
    source_data : 带标签的源域全部样本
    target_data : 目标被试全部样本（仅推断阶段用标签计算指标）

    Returns
    -------
    y_true, y_pred, y_prob  (np.ndarray)
    """
    sourcex, sourcey = self._split_xy(source_data)
    targetx, targety = self._split_xy(target_data)

    # ----- 数据预处理（复用项目逻辑） -----
    # KNN补缺失值
    sourcex_imputed = phm.impute_missing_values_by_knn(sourcex, self.cfg.knn_k)

    # SMOTE平衡数据
    sourcex_balanced, sourcey_balanced = phm.balance_data(
        sourcex_imputed, sourcey, self.cfg.smote_seed
    )

    # 标准化
    scaler = StandardScaler()
    sourcex_scaled = scaler.fit_transform(sourcex_balanced)
    targetx_scaled = scaler.transform(targetx)

    # ----- 张量（复用项目逻辑） -----
    sourcex_train = (
        torch.tensor(sourcex_scaled, dtype=torch.float32)
        .unsqueeze(2)
        .to(self.cfg.device)
    )
    sourcey_train = torch.tensor(sourcey_balanced, dtype=torch.long).to(
        self.cfg.device
    )
    targetx_train = (
        torch.tensor(targetx_scaled, dtype=torch.float32)
        .unsqueeze(2)
        .to(self.cfg.device)
    )

    # 检查数据有效性（复用项目逻辑）
    assert (
        not torch.isnan(sourcex_train).any()
        and not torch.isinf(sourcex_train).any()
    )
    assert (
        not torch.isnan(targetx_train).any()
        and not torch.isinf(targetx_train).any()
    )

    # ----- 模型（复用项目逻辑） -----
    input_dim = sourcex_train.shape[1]
    self.model_e = da.FeatureEncoderWithAttention(input_dim).to(self.cfg.device)
    self.model_c1 = da.Classifier(128, 64, 32, 3).to(self.cfg.device)
    self.model_c2 = da.Classifier(128, 256, 128, 3).to(self.cfg.device)
    self.model_d = da.DomainDiscriminator(128).to(self.cfg.device)

    opt_e = optim.Adam(self.model_e.parameters(), lr=self.cfg.lr_encoder)
    opt_c1 = optim.Adam(self.model_c1.parameters(), lr=self.cfg.lr_classifier)
    opt_c2 = optim.Adam(self.model_c2.parameters(), lr=self.cfg.lr_classifier)
    opt_d = optim.Adam(self.model_d.parameters(), lr=self.cfg.lr_domain_discriminator)

    # DataLoader（复用项目逻辑）
    source_loader = DataLoader(
        TensorDataset(sourcex_train, sourcey_train),
        batch_size=self.cfg.batch_size,
        shuffle=True,
    )
    target_loader = DataLoader(
        TensorDataset(targetx_train), batch_size=self.cfg.batch_size, shuffle=True
    )

    # 训练（复用项目的mcd_train）
    loss_mgr = LossManager()
    da.mcd_train(
        source_loader,
        target_loader,
        self.model_e,
        self.model_c1,
        self.model_c2,
        self.model_d,
        opt_e,
        opt_c1,
        opt_c2,
        opt_d,
        self.cfg,
        self.logger,
        loss_mgr,
    )

    # 推断（复用项目逻辑）
    self.model_e.eval()
    self.model_c1.eval()
    self.model_c2.eval()

    with torch.no_grad():
        features = self.model_e(targetx_train)
        pred = (self.model_c1(features) + self.model_c2(features)) / 2
        y_pred = pred.argmax(dim=1).cpu().numpy()
        y_prob = pred.cpu().numpy()

    return targety, y_pred, y_prob
  
  def _split_xy(self, data: pd.DataFrame):
     x = data.drop(columns=["subject_id", "MWL_Rating"]).values
     y = data["MWL_Rating"].values
     return x, y
        
class ReverseValidator:
  """
  外层 LOSO + 内层反向验证 选超参
  复用项目的反向验证逻辑
  """
  def __init__(self,
              base_cfg: Config,
              param_grid: dict,
              metric: str = "f1",
              n_inner_folds: int = 3,
              n_jobs: int = 1,
              logger=None):
    self.base_cfg = base_cfg
    self.param_grid = list(ParameterGrid(param_grid))
    self.metric = metric
    self.n_inner_folds = n_inner_folds
    self.n_jobs = n_jobs
    self.logger = logger

  # -------- 核心入口 --------
  def search(self, full_df: pd.DataFrame):
      history = []
      best_score, best_params = -np.inf, None

      for idx, params in enumerate(self.param_grid, 1):
          if self.logger:
              self.logger.log("[{}/{}] param={}", idx, len(self.param_grid), params, level="INFO")
          cfg = self._merge_cfg(params)

          outer_scores = self._outer_loso(full_df, cfg)
          mean_score = np.mean(outer_scores)
          history.append((params, mean_score))

          if self.logger:
              self.logger.log("  mean {}: {:.4f}", self.metric, mean_score, level="INFO")
          if mean_score > best_score:
              best_score, best_params = mean_score, params

      if self.logger:
          self.logger.log("=== BEST === param={}  {:.4f}", best_params, best_score, level="INFO")
      return self._merge_cfg(best_params), history

  # -------- 外层 LOSO --------
  def _outer_loso(self, df, cfg):
      subjects = df["subject_id"].unique()
      scores = []
      
      for tgt_subj in subjects:
          if self.logger:
              self.logger.log("Target subject: {}", tgt_subj, level="INFO")
          
          src_df = df[df["subject_id"] != tgt_subj]
          tgt_df = df[df["subject_id"] == tgt_subj]

          # ---------- 内层反向验证 ----------
          inner_best_cfg = self._inner_reverse_val(src_df, cfg)

          # ---------- 真实目标域评估 ----------
          trainer = ReverseValidateModelTrainer(inner_best_cfg, logger=self.logger)
          y_true, y_pred, y_prob = trainer.fit_predict(src_df, tgt_df)
          scores.append(self._score(y_true, y_pred, y_prob))
          
          if self.logger:
              self.logger.log("Subject {} score: {:.4f}", tgt_subj, scores[-1], level="INFO")
      
      return scores

  # -------- 内层反向验证 --------
  def _inner_reverse_val(self, src_df, cfg):
      inner_subjects = src_df["subject_id"].unique()
      kf = KFold(n_splits=self.n_inner_folds, shuffle=True, random_state=42)

      fold_scores = []
      for fold_idx, (train_idx, val_idx) in enumerate(kf.split(inner_subjects)):
          train_subj, val_subj = inner_subjects[train_idx], inner_subjects[val_idx]
          train_df = src_df[src_df["subject_id"].isin(train_subj)]
          val_df = src_df[src_df["subject_id"].isin(val_subj)]

          # 创建临时配置，减少迭代次数
          temp_cfg = copy.deepcopy(cfg)
          temp_cfg.step1_iter = 100  # 减少迭代次数
          temp_cfg.step2_iter = 3
          temp_cfg.step3_iter = 50
          temp_cfg.step4_iter = 50

          # 正向训练
          forward_trainer = ReverseValidateModelTrainer(temp_cfg, logger=None)
          y_true, y_pred, y_prob = forward_trainer.fit_predict(train_df, val_df)
          
          # 对验证集自标注
          val_x, _ = forward_trainer._split_xy(val_df)
          val_x_imputed = phm.impute_missing_values_by_knn(val_x, temp_cfg.knn_k)
          scaler = StandardScaler()
          val_x_scaled = scaler.fit_transform(val_x_imputed)
          
          # 使用正向模型进行自标注
          val_x_tensor = torch.tensor(val_x_scaled, dtype=torch.float32).unsqueeze(2).to(temp_cfg.device)
          
          # 获取正向模型的预测作为伪标签
          forward_trainer.model_e.eval()
          forward_trainer.model_c1.eval()
          forward_trainer.model_c2.eval()
          
          with torch.no_grad():
              features = forward_trainer.model_e(val_x_tensor)
              pred = (forward_trainer.model_c1(features) + forward_trainer.model_c2(features)) / 2
              pseudo_labels = pred.argmax(dim=1).cpu().numpy()
          
          # 创建带伪标签的验证集
          val_df_labeled = val_df.copy()
          val_df_labeled['MWL_Rating'] = pseudo_labels
          
          # 反向训练
          reverse_trainer = ReverseValidateModelTrainer(temp_cfg, logger=None)
          _, reverse_pred, _ = reverse_trainer.fit_predict(val_df_labeled, train_df)
          
          # 在训练集上评估反向模型
          train_x, train_y = reverse_trainer._split_xy(train_df)
          fold_score = self._score(train_y, reverse_pred, np.zeros_like(reverse_pred))  # 简化处理
          fold_scores.append(fold_score)

      # 返回原始配置（不包含减少的迭代次数）
      return cfg

  # -------- 工具 --------
  def _merge_cfg(self, params):
      cfg = copy.deepcopy(self.base_cfg)
      for k, v in params.items():
          setattr(cfg, k, v)
      return cfg

  def _score(self, y_true, y_pred, y_prob):
      if self.metric == "f1":
          return f1_score(y_true, y_pred, average="weighted", zero_division=0)
      if self.metric == "accuracy":
          return balanced_accuracy_score(y_true, y_pred)
      if self.metric == "precision":
          return precision_score(y_true, y_pred, average="weighted", zero_division=0)
      if self.metric == "recall":
          return recall_score(y_true, y_pred, average="weighted", zero_division=0)
      if self.metric == "auc":
          n_cls = len(np.unique(y_true))
          y_bin = label_binarize(y_true, classes=list(range(n_cls)))
          return roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
      raise ValueError(f"Unknown metric {self.metric}")
  

def run_hyperparameter_search():
  """
  使用新的类进行超参数搜索
  """
  
  # 1. 加载数据
  cfg = Config()
  full_df = phm.load_eeg_data(
      cfg.subjects,
      cfg.data_path,
      cfg.low_level,
      cfg.mid_level,
      cfg.high_level,
      "basic"
  )
  
  # 2. 设置日志
  logger = Logger(cfg.log_path)
  logger.log("Data shape: {}", full_df.shape, level="INFO")
  
  # 3. 定义超参数网格
  param_grid = {
      'lr_encoder': [1e-4, 1e-3, 1e-2],
      'lr_classifier': [1e-4, 1e-3, 1e-2],
      'batch_size': [32, 64],
  }
  
  # 4. 创建反向验证器
  validator = ReverseValidator(
      base_cfg=cfg,
      param_grid=param_grid,
      metric="f1",
      n_inner_folds=3,
      logger=logger
  )
  
  # 5. 搜索最佳参数
  best_cfg, history = validator.search(full_df)
  
  # 6. 使用最佳参数进行最终训练
  logger.log("Using best config: {}", best_cfg, level="INFO")
  
  # 这里可以调用你原有的run_training函数
  # performance = run_training(full_df, best_cfg, logger)
  
  return best_cfg, history
