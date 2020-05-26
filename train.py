from omegaconf import DictConfig
import hydra

from model.model import get_model

@hydra.main(config_path='config/config.yaml', strict=True)
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    model = get_model(cfg)

    print(model)
    
if __name__ == '__main__':
    main()