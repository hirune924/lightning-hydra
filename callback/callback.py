import torch
import pytorch_lightning as pl


class MyCallback(pl.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_train_start(self, trainer, pl_module):
        if self.cfg.training.scheduler.class_name == "scheduler.scheduler.CyclicLR":
            epoch_len = int(len(trainer.train_dataloader) / self.cfg.trainer.accumulate_grad_batches)
            step_size_up = int(epoch_len * 0.3)
            step_size_down = epoch_len - step_size_up
            trainer.lr_schedulers[0]["scheduler"] = torch.optim.lr_scheduler.CyclicLR(
                trainer.optimizers[0], step_size_up=step_size_up, step_size_down=step_size_down, **self.cfg.training.scheduler.params
            )
