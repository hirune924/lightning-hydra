from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils

import glob
from utils.utils import flatten_dict, load_pytorch_model
from callback.callback import MyCallback
from pytorch_lightning import Trainer, seed_everything
from model.model import get_model
from systems.system import PLRegressionImageClassificationSystem
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger

# from pytorch_lightning.logging.neptune import NeptuneLogger
from logger.logger import CustomNeptuneLogger
from pytorch_lightning import loggers

import sklearn.metrics as metrics
from metrics.metric import (
    lazy_accuracy,
    monitored_cohen_kappa_score,
)
from losses.loss import get_loss
import itertools
# For dataset
from torch.utils.data import DataLoader
from utils.utils import load_obj, preds_rounder

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

class O2UNetSystem(PLRegressionImageClassificationSystem):
    def __init__(self, hparams: DictConfig = None, model=None):
        super(O2UNetSystem, self).__init__(hparams=hparams, model=model)
        self.epoch = 0

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y, _, _, img_id, img_idx = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        log = {"train_loss": loss}
        
        return {"loss": loss, "img_id": img_id, "img_idx": img_idx, "log": log}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        img_id_list = list(itertools.chain.from_iterable([x["img_id"] for x in outputs]))
        img_idx_list = torch.cat([x["img_idx"] for x in outputs]).cpu().detach().numpy().copy()

        pd.DataFrame({'img_idx':img_idx_list, 'img_id':img_id_list}).to_csv('epoch{}_losses.csv'.format(self.epoch))
        self.epoch += 1
        log = {"avg_train_loss": avg_loss}
        return {"avg_train_loss": avg_loss, "log": log}

    # For Validation
    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y, data_provider, gleason_score, img_id, img_idx  = batch
        y_hat = self.forward(x)
        # val_loss = self.criteria(y_hat, y.view(-1, 1))
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)

        return {
            "val_loss": val_loss,
            "y": y,
            "y_hat": y_hat,
            "data_provider": data_provider,
            "gleason_score": gleason_score,
        }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        y = torch.cat([x["y"] for x in outputs]).cpu().detach().numpy().copy()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu().detach().numpy().copy()

        data_provider = torch.cat([x["data_provider"] for x in outputs]).cpu().detach().numpy().copy()
        gleason_score = torch.cat([x["gleason_score"] for x in outputs]).cpu().detach().numpy().copy()

        if self.y2pred == "round":
            preds = preds_rounder(y_hat, self.num_classes)
        elif self.y2pred == "argmax":
            preds = np.argmax(y_hat, axis=1)

        val_acc = metrics.accuracy_score(y, preds)

        val_qwk, qwk_o, qwk_e = monitored_cohen_kappa_score(y, preds, weights="quadratic", verbose=True)
        karolinska_qwk = metrics.cohen_kappa_score(y[data_provider == 0], preds[data_provider == 0], weights="quadratic", labels=range(self.num_classes),)
        radboud_qwk = metrics.cohen_kappa_score(y[data_provider == 1], preds[data_provider == 1], weights="quadratic", labels=range(self.num_classes),)
        sample_idx = (gleason_score != 0) & (gleason_score != 1) & (gleason_score != 2)
        sample_qwk = metrics.cohen_kappa_score(y[sample_idx], preds[sample_idx], weights="quadratic", labels=range(self.num_classes),)

        log = {
            "avg_val_loss": avg_loss,
            "val_acc": val_acc,
            "val_qwk": val_qwk,
            "karolinska_qwk": karolinska_qwk,
            "radboud_qwk": radboud_qwk,
            "sample_qwk": sample_qwk,
            "val_qwk_o": qwk_o,
            "val_qwk_e": qwk_e,
        }

        return {"avg_val_loss": avg_loss, "log": log}

def get_datasets(cfg: DictConfig) -> dict:
    
    cfg = OmegaConf.create(cfg)
    df = pd.read_csv(utils.to_absolute_path(os.path.join(cfg.dataset.data_dir, "train.csv")))
    
    #if cfg.cleansing is not None:
    #    del_df = pd.read_csv(utils.to_absolute_path(cfg.cleansing))
    #    for img_id in del_df['image_id']:
    #        df = df[df['image_id'] != img_id]

    #kf = load_obj(cfg.dataset.split.class_name)(**cfg.dataset.split.params)

    #for fold, (train_index, val_index) in enumerate(kf.split(df.values, df["isup_grade"].astype(str) + df["data_provider"],)):
    #    df.loc[val_index, "fold"] = int(fold)
    #df["fold"] = df["fold"].astype(int)

    #train_df = df[df["fold"] != cfg.dataset.fold]
    #valid_df = df[df["fold"] == cfg.dataset.fold]
    train_df = df[:1001]
    valid_df = df[:32]

    train_augs_conf = OmegaConf.to_container(cfg.dataset.augmentation.train, resolve=True)
    train_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in train_augs_conf]
    train_augs = A.Compose(train_augs_list)

    valid_augs_conf = OmegaConf.to_container(cfg.dataset.augmentation.valid, resolve=True)
    valid_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.dataset.augmentation.valid]
    valid_augs = A.Compose(valid_augs_list)

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
    )

    return {"train": train_dataset, "valid": valid_dataset}


class PANDADataset(Dataset):
    """PANDA Dataset."""

    def __init__(
        self, dataframe, data_dir, transform=None, load_type="png", train=True, target_type="float", K=16, auto_ws=True, window_size=128, layer=0, scale_aug=True,
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
                scale_factor = np.clip(np.random.normal(loc=2.0, scale=1.0, size=1), 0.5, 3.5,) if self.train else 2.0
            else:
                scale_factor = 2.0
            image = load_img(img_name, K=self.K, scaling_factor=scale_factor, layer=self.layer, auto_ws=self.auto_ws, window_size=self.window_size,)
        data_provider = self.data.loc[idx, "data_provider"]
        gleason_score = self.data.loc[idx, "gleason_score"]
        isup_grade = self.data.loc[idx, "isup_grade"]

        if self.transform:
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
            self.data.loc[idx, "image_id"],
            idx
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


# @hydra.main(config_path="config", strict=False)
@hydra.main(config_path="config/config.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    neptune_logger = CustomNeptuneLogger(params=flatten_dict(OmegaConf.to_container(cfg, resolve=True)), **cfg.logging.neptune_logger)
    tb_logger = loggers.TensorBoardLogger(**cfg.logging.tb_logger)

    lr_logger = LearningRateLogger()

    # TODO change to cyclicLR per epochs
    my_callback = MyCallback(cfg)

    model = get_model(cfg)
    if cfg.model.ckpt_path is not None:
        ckpt_pth = glob.glob(utils.to_absolute_path(cfg.model.ckpt_path))
        model = load_pytorch_model(ckpt_pth[0], model)

    seed_everything(2020)

    # TODO change to enable logging losses
    lit_model = O2UNetSystem(hparams=cfg, model=model)

    checkpoint_callback_conf = OmegaConf.to_container(cfg.callbacks.model_checkpoint, resolve=True)
    checkpoint_callback = ModelCheckpoint(**checkpoint_callback_conf)

    early_stop_callback_conf = OmegaConf.to_container(cfg.callbacks.early_stop, resolve=True)
    early_stop_callback = EarlyStopping(**early_stop_callback_conf)

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=[tb_logger, neptune_logger],
        # logger=[tb_logger],
        callbacks=[lr_logger, my_callback],
        **cfg.trainer
    )

    # TODO change to train with all data

    datasets = get_datasets(OmegaConf.to_container(cfg, resolve=True))
    train_dataset = datasets["train"]
    valid_dataset = datasets["valid"]
    trainer.fit(lit_model, 
    train_dataloader=DataLoader(train_dataset, **cfg["training"]["dataloader"]["train"]), 
    val_dataloaders=DataLoader(valid_dataset, **cfg["training"]["dataloader"]["valid"]))

    # trainer.test()


if __name__ == "__main__":
    main()
