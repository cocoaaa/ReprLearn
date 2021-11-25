from typing import Any
from pathlib import Path
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class DebugCallback(Callback):

    def __init__(self, log_every: int, num_samples: int, save_dir: Path) -> None:
        """ At every `log_every` epoch, use the current G to generate a
        `num_samples` number of datapts.
        Also, use the current D to compute the avg. score on the sample of G.
        G much have a method called ".sample"""
        self.log_every = log_every
        self.num_samples = num_samples
        self.save_dir = save_dir
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            print(f"Created dir for samples: {save_dir}")

    def on_train_start(self, trainer, pl_module) -> None:
        print("Train started")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch < 1:
            print("Train epoch 0 started")
            print("device: ", pl_module.device)

    def on_train_batch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        """Called when the train batch begins."""
        if trainer.current_epoch < 1 and batch_idx < 1:
            print(f"Saving real images from first batch at ep {trainer.current_epoch}, batch {batch_idx}...")
            x_real = batch['x'][:self.num_samples]  # (64, nc, h, w) tensor
            out_fp = self.save_dir / 'x_real.png'
            torchvision.utils.save_image(x_real, out_fp)
            print("Saved!")

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with torch.no_grad():
            was_training = pl_module.training
            pl_module.eval()
            x_gen = pl_module.sample(self.num_samples, pl_module.device)
            pl_module.train(was_training)

            fp = self.save_dir / f"x_gen_epoch={trainer.current_epoch}.png"
            torchvision.utils.save_image(x_gen, fp)



