#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.utils import load_obj

import torch
import torch.nn as nn

def get_optimizer(model_params, cfg):
    optimizer = load_obj(cfg.training.optimizer.class_name)
    optimizer = optimizer(model_params, **cfg.training.optimizer.params)

    return optimizer


