import torch
import torch.nn.functional as F
from config import Config
import torch.nn as nn


def gaussian_kernel(x, y, cfg: Config, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(x.size()[0]) + int(y.size()[0])
    total = torch.cat([x, y], dim=0).to(cfg.device)

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))

    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth = max(bandwidth.item(), 1e-6)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_tmp) for bandwidth_tmp in bandwidth_list]

    return sum(kernel_val)


class LossManager:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    @staticmethod
    def step1_loss(out1, out2, y):
        # 默认使用交叉熵
        return F.cross_entropy(out1, y) + F.cross_entropy(out2, y)

    # @staticmethod
    # def discrepancy_loss(out1, out2):
    #     return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

    def double_classifier_loss_mse(self, out1, out2, y):
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        p1 = torch.softmax(out1, dim=1)
        p2 = torch.softmax(out2, dim=1)
        return F.mse_loss(p1, y_onehot) + F.mse_loss(p2, y_onehot)

    @staticmethod
    def double_classifier_loss_ce(p1, p2, y_true):
        return F.cross_entropy(p1, y_true) + F.cross_entropy(p2, y_true)

    def mmd_loss(self, source, target, cfg):
        batch_size = int(source.size()[0])
        kernels = gaussian_kernel(source, target, cfg)
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]

        # return torch.mean(xx + yy - xy - yx)
        return torch.mean(xx) + torch.mean(yy) - torch.mean(xy) - torch.mean(yx)
    
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

    def discrepancy_stable(self, out1, out2, eps=1e-8):
        p1 = F.softmax(out1, dim=1).clamp(min=eps)
        p2 = F.softmax(out2, dim=1).clamp(min=eps)
        return torch.mean(torch.abs(p1 - p2))
    
    def domain_loss(self, domain_preds, domain_labels):
        return self.ce_loss(domain_preds, domain_labels)
