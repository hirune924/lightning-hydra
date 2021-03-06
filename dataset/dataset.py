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
import numpy as np
from hydra import utils

from utils.resize_intl_tile import load_img


def get_datasets(cfg: DictConfig) -> dict:

    cfg = OmegaConf.create(cfg)
    df = pd.read_csv(utils.to_absolute_path(os.path.join(cfg.dataset.data_dir, "train.csv")))

    #if cfg.dataset.cleansing is not None:
    #    del_df = pd.read_csv(utils.to_absolute_path(cfg.dataset.cleansing))
    #    for img_id in del_df['image_id']:
    #        df = df[df['image_id'] != img_id]
    #    df = df.reset_index(drop=True)

    kf = load_obj(cfg.dataset.split.class_name)(**cfg.dataset.split.params)

    for fold, (train_index, val_index) in enumerate(kf.split(df.values, df["isup_grade"].astype(str) + df["data_provider"],)):
        df.loc[val_index, "fold"] = int(fold)
    df["fold"] = df["fold"].astype(int)

    train_df = df[df["fold"] != cfg.dataset.fold]
    valid_df = df[df["fold"] == cfg.dataset.fold]

    if cfg.dataset.cleansing is not None:
        del_df = pd.read_csv(utils.to_absolute_path(cfg.dataset.cleansing))
        #del_df = del_df[del_df['data_provider'] == 'radboud'] ## critical change
        train_df = train_df[~train_df['image_id'].isin(del_df['image_id'])]


    if cfg.dataset.drop is not None:
        train_df = train_df[train_df["data_provider"] != cfg.dataset.drop]

    train_augs_conf = OmegaConf.to_container(cfg.dataset.augmentation.train, resolve=True)
    train_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in train_augs_conf]
    train_augs = A.Compose(train_augs_list)

    valid_augs_conf = OmegaConf.to_container(cfg.dataset.augmentation.valid, resolve=True)
    valid_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.dataset.augmentation.valid]
    valid_augs = A.Compose(valid_augs_list)

    hard_aug = None
    if cfg.dataset.hard_aug is not None:
        hard_aug_conf = OmegaConf.to_container(cfg.dataset.hard_aug, resolve=True)
        hard_aug_list = [load_obj(i["class_name"])(**i["params"]) for i in hard_aug_conf]
        hard_aug = A.Compose(hard_aug_list)

    train_dataset = PANDADataset(
        train_df,
        cfg.dataset.data_dir,
        transform=train_augs,
        load_type=cfg.dataset.load_type,
        train=True,
        target_type=cfg.dataset.target_type,
        K=cfg.dataset.K,
        auto_ws=cfg.dataset.auto_ws,
        window_size=cfg.dataset.window_size,
        layer=cfg.dataset.layer,
        scale_aug=cfg.dataset.scale_aug,
        aug_mean= cfg.dataset.aug_mean,
        aug_scale= cfg.dataset.aug_scale,
        hard_aug=hard_aug,
        mixtile=cfg.dataset.mixtile
    )

    valid_dataset = PANDADataset(
        valid_df,
        cfg.dataset.data_dir,
        transform=valid_augs,
        load_type=cfg.dataset.load_type,
        train=False,
        target_type=cfg.dataset.target_type,
        K=cfg.dataset.K,
        auto_ws=cfg.dataset.auto_ws,
        window_size=cfg.dataset.window_size,
        layer=cfg.dataset.layer,
        scale_aug=cfg.dataset.scale_aug,
        aug_mean= cfg.dataset.aug_mean,
        aug_scale= cfg.dataset.aug_scale,
        hard_aug=None,
    )

    return {"train": train_dataset, "valid": valid_dataset}


class PANDADataset(Dataset):
    """PANDA Dataset."""

    def __init__(
        self, dataframe, data_dir, transform=None, load_type="png", train=True, target_type="float", K=16, auto_ws=True, window_size=128, layer=0, scale_aug=True, aug_mean=2.0, aug_scale=1.0, hard_aug=None, mixtile=None,
    ):
        """
        Args:
            data_path (string): data path(glob_pattern) for dataset images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataframe.reset_index(drop=True)  # pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
        self.transform = transform
        self.data_dir = data_dir
        self.load_type = load_type
        self.train = train
        self.target_type = target_type
        self.auto_ws = auto_ws
        self.window_size = window_size
        self.layer = layer
        self.scale_aug = scale_aug
        self.K = K
        self.aug_mean = aug_mean
        self.aug_scale = aug_scale
        self.hard_aug = hard_aug

        self.mixtile = mixtile
        if self.mixtile is not None:
            self.radboud_cache = {0: np.full((2048, 2048, 3), 255, dtype=np.uint8),
             3: np.full((2048, 2048, 3), 255, dtype=np.uint8),
             4: np.full((2048, 2048, 3), 255, dtype=np.uint8),
             5: np.full((2048, 2048, 3), 255, dtype=np.uint8)}
            self.karolinska_cache = {0: np.full((2048, 2048, 3), 255, dtype=np.uint8),
             3: np.full((2048, 2048, 3), 255, dtype=np.uint8),
             4: np.full((2048, 2048, 3), 255, dtype=np.uint8),
             5: np.full((2048, 2048, 3), 255, dtype=np.uint8)}
            self.work = np.full((2048, 2048, 3), 255, dtype=np.uint8)
            self.gl_dict = {"negative": 0, "0+0": 0, "3+3": 3, "4+4": 4, "5+5": 5}



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.load_type == "png":
            img_name = utils.to_absolute_path(os.path.join(os.path.join(self.data_dir, "train_images/"), self.data.loc[idx, "image_id"] + "." + "png",))
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.load_type == "tiff_tile":
            img_name = utils.to_absolute_path(os.path.join(os.path.join(self.data_dir, "train_images/"), self.data.loc[idx, "image_id"] + "." + "tiff",))
            if self.scale_aug:
                scale_factor = np.clip(np.random.normal(loc=self.aug_mean, scale=self.aug_scale, size=1), 0.5, 2 * self.aug_mean - 0.5,) if self.train else self.aug_mean
            else:
                scale_factor = self.aug_mean
            image = load_img(img_name, K=self.K, scaling_factor=scale_factor, layer=self.layer, auto_ws=self.auto_ws, window_size=self.window_size,)
        data_provider = self.data.loc[idx, "data_provider"]
        gleason_score = self.data.loc[idx, "gleason_score"]
        isup_grade = self.data.loc[idx, "isup_grade"]

        if self.mixtile is not None:
            if gleason_score in ["negative", "0+0", "3+3", "4+4", "5+5"]:
                p = np.random.rand()
                if p > 0.5:
                    n_tile = np.random.randint(1,5)
                    target_tile = np.random.choice(np.arange(16), size=n_tile, replace=False)
                    target_gleason = np.random.choice([3,4,5]) if gleason_score not in ["negative", "0+0"] else 0
                    image = cv2.resize(image, (2048, 2048))
                    self.work = image.copy()
                    for region in target_tile:
                        x = region // 4
                        y = region % 4
                        if data_provider == 'radboud':
                            self.work[x * 512: (x + 1) * 512, y * 512: (y + 1) * 512, :] = self.radboud_cache[target_gleason][x * 512: (x + 1) * 512, y * 512: (y + 1) * 512, :]
                        elif data_provider == 'karolinska':
                            self.work[x * 512: (x + 1) * 512, y * 512: (y + 1) * 512, :] = self.karolinska_cache[target_gleason][x * 512: (x + 1) * 512, y * 512: (y + 1) * 512, :]
                    if data_provider == 'radboud':
                        self.radboud_cache[self.gl_dict[gleason_score]] = image.copy()
                    elif data_provider == 'karolinska':
                        self.karolinska_cache[self.gl_dict[gleason_score]] = image.copy()
                    image = self.work.copy()
                    isup_grade = gleason2isup('{}+{}'.format(self.gl_dict[gleason_score], target_gleason))



        if self.hard_aug is None:
            if self.transform:
                image = self.transform(image=image)
                image = torch.from_numpy(image["image"].transpose(2, 0, 1))
        else:
            if isup_grade in [4, 5]:
                image = self.hard_aug(image=image)
                image = torch.from_numpy(image["image"].transpose(2, 0, 1))
            else:
                image = self.transform(image=image)
                image = torch.from_numpy(image["image"].transpose(2, 0, 1))                           

        if self.target_type == "float":
            isup_grade = torch.Tensor([isup_grade]).float()
        elif self.target_type == "long":
            isup_grade = isup_grade
        return (
            image,
            isup_grade,
            data_provider2id(data_provider),
            gleason2id(gleason_score),
        )


def data_provider2id(data_provider):
    trans_dict = {"karolinska": 0, "radboud": 1}
    return trans_dict[data_provider]


def gleason2id(gleason):
    trans_dict = {
        "negative": 0,
        "0+0": 1,
        "3+3": 2,
        "3+4": 3,
        "4+3": 4,
        "4+4": 5,
        "4+5": 6,
        "5+4": 7,
        "5+5": 8,
        "3+5": 9,
        "5+3": 10,
    }
    return trans_dict[gleason]

def gleason2isup(gleason):
    if gleason in [ '0+0', 'negative' ]:
        return 0
    elif gleason == '3+3':
        return 1
    elif gleason == '3+4':
        return 2
    elif gleason == '4+3':
        return 3
    elif gleason in [ '4+4', '3+5', '5+3' ]:
        return 4
    elif gleason in [ '4+5', '5+4', '5+5' ]:
        return 5
    else:
        pass
    return 0