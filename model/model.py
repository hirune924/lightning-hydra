from utils.utils import load_obj

import torch
import torch.nn as nn
import torchvision.models as models
import pretrainedmodels
import segmentation_models_pytorch as smp
import timm

from layer.layer import AdaptiveConcatPool2d, GeM
from utils.utils import load_pytorch_model

import glob
from hydra import utils

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def get_model(cfg):
    model = load_obj(cfg.model.class_name)
    model = model(**cfg.model.params)

    return model


def resnet18(pretrained=True, num_classes=1000):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def se_resnet50(
    pretrained="imagenet", num_classes=1000, pool="avg", pool_size=1,
):
    # model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
    model = pretrainedmodels.__dict__["se_resnet50"](num_classes=1000, pretrained=None)
    if pretrained is not None:
        ckpt_pth = glob.glob(utils.to_absolute_path(pretrained))
        model.load_state_dict(torch.load(ckpt_pth[0]))

    in_features = model.last_linear.in_features
    if pool == "avg":
        model.last_linear = nn.Linear(in_features * (pool_size ** 2), num_classes)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(pool_size)
    elif pool == "avgmax":
        model.last_linear = nn.Linear(in_features * 2 * (pool_size ** 2), num_classes)
        model.avg_pool = AdaptiveConcatPool2d(pool_size)
    elif pool == "gem":
        model.last_linear = nn.Linear(in_features, num_classes)
        model.avg_pool = GeM()
    return model


def se_net(
    model_name="se_resnet50", pretrained="imagenet", num_classes=1000, pool="avg", pool_size=1,
):
    # model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
    if pretrained == 'imagenet':
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    elif pretrained is not None :
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
        ckpt_pth = glob.glob(utils.to_absolute_path(pretrained))
        model.load_state_dict(torch.load(ckpt_pth[0]))
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
    

    in_features = model.last_linear.in_features
    if pool == "avg":
        model.last_linear = nn.Linear(in_features * (pool_size ** 2), num_classes)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(pool_size)
    elif pool == "avgmax":
        model.last_linear = nn.Linear(in_features * 2 * (pool_size ** 2), num_classes)
        model.avg_pool = AdaptiveConcatPool2d(pool_size)
    elif pool == "gem":
        model.last_linear = nn.Linear(in_features, num_classes)
        model.avg_pool = GeM()
    return model

def timm_custom(model_name='gluon_seresnext50_32x4d', num_classes=1, pretrained=None, pool_size=1, pool_type='avg', head_type='linear'):
    if pretrained is not None :
        model = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
        ckpt_pth = glob.glob(utils.to_absolute_path(pretrained))
        model = load_pytorch_model(ckpt_pth[0], model)
    else:
        model = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=False)

    if pool_type == 'avg':
        model.global_pool = nn.AdaptiveAvgPool2d(pool_size)
    elif pool_type == 'avgdrop':
        model.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(pool_size),
                                          nn.Dropout2d(p=0.3, inplace=False))
    if head_type=='linear':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features*pool_size*pool_size, num_classes)
    elif head_type=='custom':
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_features*pool_size*pool_size, 512), nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False),
                                 nn.BatchNorm1d(512),nn.Linear(512,num_classes))
    return model
