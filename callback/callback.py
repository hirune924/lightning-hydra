import torch
import pytorch_lightning as pl

class MyCallback(pl.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_train_start(self, trainer, pl_module):
        if self.cfg.training.scheduler.class_name == 'scheduler.scheduler.CyclicLR':
            step_size_up = int(len(trainer.train_dataloader)/2)
            step_size_down = len(trainer.train_dataloader) - step_size_up
            trainer.lr_schedulers[0]['scheduler'] = torch.optim.lr_scheduler.CyclicLR(
                trainer.optimizers[0],
                step_size_up=step_size_up, 
                step_size_down=step_size_down, **self.cfg.training.scheduler.params)
