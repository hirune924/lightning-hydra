from omegaconf import DictConfig, OmegaConf
import hydra

from pytorch_lightning import Trainer, seed_everything
from model.model import get_model
from systems.system import PLRegressionImageClassificationSystem
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping


@hydra.main(config_path='config/config.yaml', strict=True)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    model = get_model(cfg)

    seed_everything(2020)

    lit_model = PLRegressionImageClassificationSystem(hparams=cfg, model=model)
    
    checkpoint_callback_conf = OmegaConf.to_container(cfg.callbacks.model_checkpoint, resolve=True)
    checkpoint_callback = ModelCheckpoint(**checkpoint_callback_conf)

    early_stop_callback_conf = OmegaConf.to_container(cfg.callbacks.early_stop, resolve=True)
    early_stop_callback = EarlyStopping(**early_stop_callback_conf)

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        **cfg.trainer)

    trainer.fit(lit_model)

if __name__ == '__main__':
    main()