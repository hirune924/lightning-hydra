#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.utils import load_obj

import torch
import torch.nn as nn

try:
    import robust_loss_pytorch
    _ROBUST_LOSS_PYTORCH_AVAILABLE = True
except ImportError:
    _ROBUST_LOSS_PYTORCH_AVAILABLE = False


def get_loss(cfg):
    loss = load_obj(cfg.training.loss.class_name)
    loss = loss(**cfg.training.loss.params)

    return loss


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        criterion = nn.MSELoss(reduction=self.reduction)
        loss = torch.sqrt(criterion(x, y))
        return loss

class AdaptiveLossFunction(torch.nn.Module):
    def __init__(self):
        super(AdaptiveLossFunction, self).__init__()
        self.adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims = 1, float_dtype=torch.float32, device='cuda:0')

    def forward(self, x, y):
        loss = torch.mean(self.adaptive.lossfun((x - y)[:,None].squeeze(dim=1)))
        return loss