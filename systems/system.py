#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import albumentations as A
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import numpy as np

from utils.utils import load_obj, preds_rounder
from losses.loss import get_loss
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler
from dataset.dataset import get_datasets
from metrics.metric import (
    lazy_accuracy,
    monitored_cohen_kappa_score,
)

from omegaconf import DictConfig, OmegaConf

from argparse import Namespace

# from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


class PLRegressionImageClassificationSystem(pl.LightningModule):
    def __init__(self, hparams: DictConfig = None, model=None):
        # def __init__(self, train_loader, val_loader, model):
        super(PLRegressionImageClassificationSystem, self).__init__()
        self.hparams = OmegaConf.to_container(hparams, resolve=True)
        self.model = model
        self.num_classes = hparams.training.num_classes
        self.criteria = get_loss(hparams)
        self.y2pred = hparams.training.y2pred

    def forward(self, x):
        return self.model(x)

    # For Training
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y, _, _ = batch
        y_hat = self.forward(x)
        if self.hparams["training"]["label_mode"] == 'reverse':
            y = 5 - y
        elif self.hparams["training"]["label_mode"] == 'slide':
            y = y - 2.5
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        log = {"avg_train_loss": avg_loss}
        return {"avg_train_loss": avg_loss, "log": log}

    def configure_optimizers(self):
        # REQUIRED
        if self.hparams["training"]["loss"]["class_name"] == 'losses.loss.AdaptiveLossFunction':
            optimizer = get_optimizer(list(self.model.parameters())+list(self.criteria.adaptive.parameters()), self.hparams)
        else:    
            optimizer = get_optimizer(self.model.parameters(), self.hparams)

        scheduler = get_scheduler(optimizer, self.hparams)
        return (
            [optimizer],
            [{"scheduler": scheduler, "monitor": "avg_val_loss", "interval": self.hparams["training"]["scheduler"]["interval"],}],
        )

    def optimizer_step(
        self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
    ):
        optimizer.step()
        optimizer.zero_grad()

    # For Validation
    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y, data_provider, gleason_score = batch
        y_hat = self.forward(x)
        # val_loss = self.criteria(y_hat, y.view(-1, 1))
        if self.hparams["training"]["label_mode"] == 'reverse':
            y = 5 - y
        elif self.hparams["training"]["label_mode"] == 'slide':
            y = y - 2.5
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        if self.hparams["training"]["label_mode"] == 'reverse':
            y_hat = 5 - y_hat
            y = 5 - y
        elif self.hparams["training"]["label_mode"] == 'slide':
            y_hat = y_hat + 2.5
            y = y + 2.5
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

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y, data_provider, gleason_score = batch
        y_hat = self.forward(x)

        test_loss = self.criteria(y_hat, y.view(-1, 1).float())
        test_loss = test_loss.unsqueeze(dim=-1)

        return {
            "test_loss": test_loss,
            "y": y,
            "y_hat": y_hat,
        }

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        y = torch.cat([x["y"] for x in outputs]).cpu().detach().numpy().copy()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu().detach().numpy().copy()

        # preds = np.argmax(y_hat, axis=1)
        preds = preds_rounder(y_hat, self.num_classes)
        test_acc = metrics.accuracy_score(y, preds)
        test_qwk = metrics.cohen_kappa_score(y, preds, weights="quadratic")

        fig, ax = plt.subplots(figsize=(16, 12))
        # plot_confusion_matrix(y, preds, ax=ax)
        self.logger.experiment[1].log_image("confusion_matrix", fig)

        log = {
            "avg_test_loss": avg_loss,
            "test_acc": test_acc,
            "test_qwk": test_qwk,
        }
        return {"avg_test_loss": avg_loss, "log": log}

    # For Data
    def prepare_data(self):
        datasets = get_datasets(self.hparams)
        self.train_dataset = datasets["train"]
        self.valid_dataset = datasets["valid"]

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, **self.hparams["training"]["dataloader"]["train"])

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.valid_dataset, **self.hparams["training"]["dataloader"]["valid"])

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.valid_dataset, **self.hparams["training"]["dataloader"]["valid"])
