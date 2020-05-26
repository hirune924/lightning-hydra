from omegaconf import DictConfig
import hydra

from pytorch_lightning import Trainer
from model.model import get_model
from systems.system import PLRegressionImageClassificationSystem


@hydra.main(config_path='config/config.yaml', strict=True)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    model = get_model(cfg)

    lit_model = PLRegressionImageClassificationSystem(hparams=cfg, model=model)
    
    trainer = Trainer(**cfg.trainer)

    trainer.fit(lit_model)
    
if __name__ == '__main__':
    main()