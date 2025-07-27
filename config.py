from dataclasses import dataclass, field
from typing import List
import torch
import os

@dataclass
class Config:
  # path
  data_path: str = r"D:\Data\Group\2-nuclear_data\deeplearning"
  log_path: str = os.path.join(data_path, "logs")
  subjects: List[str] = field(default_factory=lambda: [
    'NP03', 'NP04', 'NP05', 
    'NP06', 'NP07', 'NP08', 
    'NP09', 'NP10', 'NP11', 
    'NP12', 'NP13', 'NP14', 
    'NP15', 'NP16', 'NP17', 
    'NP18', 'NP19', 'NP20', 
    'NP21', 'NP22', 'NP23', 
    'NP24', 'NP25', 'NP26',
    'NP27', 'NP28', 'NP29', 
    'NP30', 'NP31', 'NP32'])
  
  # data
  knn_k: int = 5
  smote_seed: int = 42

  # training hyper-params
  batch_size: int = 128
  max_epochs: int = 40
  lr_encoder: float = 1e-3
  lr_classifier: float = 1e-3
  lr_domain_discriminator: float = 9e-4
  clip_grad: float = 5.0

  # MCD iterations
  step1_iter: int = 1
  step2_iter: int = 4
  step3_iter: int = 1
  # step4_iter: int = 1
  lambda_GRL: float = 0.3

  # mwl level
  low_level = 1
  mid_level = 5
  high_level = 9

  # 任务定义
  num_classes: int = 3
  binary_threshold: int = 6

  # misc
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
