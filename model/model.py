from utils.utils import load_obj

import torch
import torch.nn as nn
import torchvision.models as models
import pretrainedmodels
import segmentation_models_pytorch as smp

from layer.layer import AdaptiveConcatPool2d

import glob
from hydra import utils

def get_model(cfg):
    model = load_obj(cfg.model.class_name)
    model = model(**cfg.model.params)

    return model

def resnet18(pretrained=True, num_classes=1000):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def se_resnet50(pretrained='imagenet', num_classes=1000, pool='avg', pool_size=1):
    #model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
    model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=None)
    ckpt_pth = glob.glob(utils.to_absolute_path(pretrained))
    model.load_state_dict(torch.load(ckpt_pth[0]))

    in_features = model.last_linear.in_features
    if pool=='avg':
        model.last_linear = nn.Linear(in_features*(pool_size**2), num_classes)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(pool_size)
    elif pool=='avgmax':
        model.last_linear = nn.Linear(in_features*2*(pool_size**2), num_classes)
        model.avg_pool = AdaptiveConcatPool2d(pool_size)  

    return model