#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.utils import load_obj

import torch
import torch.nn as nn

def get_scheduler(optimizer, cfg):
    scheduler = load_obj(cfg.training.scheduler.class_name)
    scheduler = scheduler(optimizer, **cfg.training.scheduler.params)

    return scheduler


