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

from omegaconf import DictConfig, OmegaConf

from argparse import Namespace


class PLRegressionImageClassificationSystem(pl.LightningModule):
    
    def __init__(self, hparams: DictConfig = None, model = None):
    #def __init__(self, train_loader, val_loader, model):
        super(PLRegressionImageClassificationSystem, self).__init__()
        self.hparams = OmegaConf.to_container(hparams, resolve=True)
        self.model = model
        self.criteria = get_loss(hparams)

    def forward(self, x):
        return self.model(x)
    
# For Training
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y.view(-1, 1).float())
        loss = loss.unsqueeze(dim=-1)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        log = {'avg_train_loss': avg_loss}
        return {'avg_train_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = get_optimizer(self.model.parameters(), self.hparams)

        scheduler = get_scheduler(optimizer, self.hparams)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'avg_val_loss'}]

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                    second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
    
# For Validation
    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criteria(y_hat, y.view(-1, 1).float())
        val_loss = val_loss.unsqueeze(dim=-1)

        return {'val_loss': val_loss, 'y': y, 'y_hat': y_hat}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        
        y = torch.cat([x['y'] for x in outputs]).cpu().detach().numpy().copy()
        y_hat = torch.cat([x['y_hat'] for x in outputs]).cpu().detach().numpy().copy()

        #preds = np.argmax(y_hat, axis=1)
        preds = preds_rounder(y_hat, self.hparams['training']['num_classes'])
        val_acc = metrics.accuracy_score(y, preds)
        val_qwk = metrics.cohen_kappa_score(y, preds, weights='quadratic')


        log = {'avg_val_loss': avg_loss, 'val_acc': val_acc, 'val_qwk': val_qwk}
        return {'avg_val_loss': avg_loss, 'log': log}

# For Data
    def prepare_data(self):
        datasets = get_datasets(self.hparams)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, **self.hparams['training']['dataloader']['train'])

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.valid_dataset, **self.hparams['training']['dataloader']['valid'])

    #def test_dataloader(self):
    #    # OPTIONAL
    #    pass


