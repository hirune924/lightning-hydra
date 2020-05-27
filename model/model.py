from utils.utils import load_obj

import torch
import torch.nn as nn
import torchvision.models as models
import pretrainedmodels
import segmentation_models_pytorch as smp



def get_model(cfg):
    model = load_obj(cfg.model.class_name)
    model = model(**cfg.model.params)

    return model

def resnet18(pretrained=True, num_classes=1000):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model