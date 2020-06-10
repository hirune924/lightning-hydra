#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.utils import load_obj

import torch
import torch.nn as nn


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
