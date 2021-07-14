import itertools
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.utils.scheduler import frange_cycle_linear
from argparse import ArgumentParser


class BetaScheduler(Callback):

    def __init__(self, n_iter, start=0.0,
                 stop=1.0, n_cycle=4, ratio=0.5,
                 log_tag: str = 'train/beta'):
        self.beta_iter = itertools.cycle(
            frange_cycle_linear(n_iter, start, stop, n_cycle, ratio)
        )
        self.log_tag = log_tag

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        beta = next(self.beta_iter)
        setattr(pl_module, 'kld_weight', beta)
        trainer.logger.log_metrics({self.log_tag: pl_module.kld_weight},
                                   step=trainer.global_step)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Add boolean argument switches: https://stackoverflow.com/a/31347222
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--use_beta_scheduler', dest='use_beta_scheduler', action='store_true')
        group.add_argument('--not_use_beta_scheduler', dest='use_beta_scheduler', action='store_false')
        parser.set_defaults(use_beta_scheduler=True)

        parser.add_argument('--beta_start', type=float, default=0.0)
        parser.add_argument('--beta_stop', type=float, default=1.0)
        parser.add_argument('--beta_n_cycle', type=int, default=4)
        parser.add_argument('--beta_ratio', type=float, default=0.5)
        parser.add_argument('--beta_log_tag', type=str, default="train/beta")

        return parser