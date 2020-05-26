#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils.utils import load_obj

import torch
from torch.utils.data import Dataset
import skimage.io
import os
import cv2
import albumentations as A
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from hydra import utils


def get_datasets(cfg: DictConfig) -> dict:
    """
    Get datases for modelling

    Args:
        cfg: config

    Returns:

    """

    cfg = OmegaConf.create(cfg)
    df = pd.read_csv(utils.to_absolute_path(os.path.join(cfg.dataset.data_dir,'train.csv')))

    kf = load_obj(cfg.dataset.split.class_name)(**cfg.dataset.split.params)

    for fold, (train_index, val_index) in enumerate(kf.split(df.values, df['isup_grade'])):
        df.loc[val_index, 'fold'] = int(fold)
    df['fold'] = df['fold'].astype(int)

    train_df = df[df['fold']!=cfg.dataset.fold]
    valid_df = df[df['fold']==cfg.dataset.fold]

    # for debug run
    if cfg.training.debug:
        train_df = train_df[:10]
        valid_df = valid_df[:10]

    train_augs_conf = OmegaConf.to_container(cfg.dataset.augmentation.train, resolve=True)
    train_augs_list = [load_obj(i['class_name'])(**i['params']) for i in train_augs_conf]
    train_augs = A.Compose(train_augs_list)

    valid_augs_conf = OmegaConf.to_container(cfg.dataset.augmentation.valid, resolve=True)
    valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg.dataset.augmentation.valid]
    valid_augs = A.Compose(valid_augs_list)

    train_dataset = PANDADataset(train_df,
                                  cfg.dataset.data_dir,
                                  transform=train_augs)

    valid_dataset = PANDADataset(valid_df,
                                  cfg.dataset.data_dir,
                                  transform=valid_augs)

    return {'train': train_dataset, 'valid': valid_dataset}


class PANDADataset(Dataset):
    """PANDA Dataset."""
    
    def __init__(self, dataframe, data_dir, transform=None):
        """
        Args:
            data_path (string): data path(glob_pattern) for dataset images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataframe.reset_index(drop=True) #pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = utils.to_absolute_path(os.path.join(os.path.join(self.data_dir, 'train_images/'), self.data.loc[idx, 'image_id'] + '.' +'png'))
        data_provider = self.data.loc[idx, 'data_provider']
        gleason_score = self.data.loc[idx, 'gleason_score']
        isup_grade = label = self.data.loc[idx, 'isup_grade']
        
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)
            image = torch.from_numpy(image['image'].transpose(2, 0, 1))
        return image, isup_grade
