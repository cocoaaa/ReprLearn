from typing import Any
from pathlib import Path
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from IPython.core.debugger import set_trace

class Log2DiskCallback(Callback):

    def __init__(self, log_every: int, num_samples: int, save_dir: Path) -> None:
        """ At every `log_every` epoch, use the current G to generate a
        `num_samples` number of datapts, and write the sample to disk (`save_dir`)
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
        if trainer.current_epoch == 0 and batch_idx == 0:
            print(f"Saving real images from first batch at ep {trainer.current_epoch}, batch {batch_idx}...")
            x_real = batch['x'][:self.num_samples]  # (64, nc, h, w) tensor
            out_fp = self.save_dir / 'x_real.png'
            torchvision.utils.save_image(x_real, out_fp)
            print("Saved snapshot of real images!")

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if outputs['loss'] is None:
    #         print('train loss_G is None!!')
    #         set_trace()

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save samples generated by current G's state to `self.save_dir` """
        if (trainer.current_epoch + 1) % self.log_every == 0:
            with torch.no_grad():
                #----
                # todo: this way of returning back to original module's training state can benefit from the context manager
                was_training = pl_module.training
                pl_module.eval()
                x_gen = pl_module.sample(self.num_samples, pl_module.device)
                pl_module.train(was_training)
                #----
                fp = self.save_dir / f"x_gen_epoch={trainer.current_epoch}_gstep={trainer.global_step}.png"
                torchvision.utils.save_image(x_gen, fp)


class Log2TBCallback(Callback):

    def __init__(self, log_every: int, num_samples: int) -> None:
        """ At every `log_every` epoch, use the current G to generate a
        `num_samples` number of datapts.
        Also, use the current D to compute the avg. score on the sample of G.
        G much have a method called ".sample"""
        self.log_every = log_every
        self.num_samples = num_samples

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with torch.no_grad():

            #----- todo: use context manager
            was_training = pl_module.training
            # print('*****') #debugF
            # print(f'on_epoch_end -- module.training: {was_training} <-- should be eval?')
            pl_module.eval()
            x_gen = pl_module.sample(self.num_samples, pl_module.device)
            pl_module.train(was_training)
            # ----- end: wrapped by context manager

            grid = torchvision.utils.make_grid(x_gen)
            trainer.logger.experiment.add_image( f"x_gen", grid, trainer.global_step)  # todo: ep should be set properly
