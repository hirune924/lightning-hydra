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

import torch.nn as nn


# @hydra.main(config_path="config", strict=False)
@hydra.main(config_path="config/config.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    neptune_logger = CustomNeptuneLogger(params=flatten_dict(OmegaConf.to_container(cfg, resolve=True)), **cfg.logging.neptune_logger)
    tb_logger = loggers.TensorBoardLogger(**cfg.logging.tb_logger)

    lr_logger = LearningRateLogger()

    my_callback = MyCallback(cfg)

    model = get_model(cfg)
    if cfg.model.ckpt_path is not None:
        ckpt_pth = glob.glob(utils.to_absolute_path(cfg.model.ckpt_path))
        model = load_pytorch_model(ckpt_pth[0], model)
    if cfg.trainer.distributed_backend == 'ddp':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    seed_everything(2020)

    lit_model = PLRegressionImageClassificationSystem(hparams=cfg, model=model)

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

    trainer.fit(lit_model)

    # trainer.test()


if __name__ == "__main__":
    main()
