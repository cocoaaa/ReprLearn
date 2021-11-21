"""
trainer_main.py

Required args:
    --model_name: "vae" or "iwae"
    --data_name: "maptiles" or "mnist"
    --latent_dim: int, eg. 10

Optional args: (partial)
    --hidden_dims: eg. --hidden_dims 32 64 128 256 (which is default)

To run:
cd scripts

nohup python train_gm.py --model_name="vae" --data_name="mnist" --latent_dim=10

nohup python train_gm.py --model_name beta_vae --kld_weight 70.0 \
--enc_type conv --dec_type conv \
--hidden_dims 32 64 128 256 --latent_dim 10 \
--data_name mnist  \
--gpu_id=2 --max_epochs=100 --batch_size=32 -lr 1e-3  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/test/" >> log.txt &

nohup python train_gm.py --model_name="iwae" --data_name="mnist" --latent_dim=10


# Train a gan with conv generator and fc discriminator
# log stdout and stdout to the log file while outputing them to the stream (`tee`)
python train_gm.py --model_name conv_fc_gan --data_name mnist \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name mnist \
--gpu_id 1 --max_epochs 300 --batch_size 1028 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/test/11-19-2021 \
2>&1 | tee log-"$(date +%F-%H:%M)".txt
"""
import argparse
import os,sys
import re
import math
from datetime import datetime
import time
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import warnings
from pprint import pprint

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers

# plmodules
from reprlearn.models.plmodules import ThreeFCs, VanillaVAE, BetaVAE, IWAE

# datamodules
from reprlearn.data.datamodules import MNISTDataModule, MaptilesDataModule

# callbacks
from reprlearn.callbacks.recon_logger import ReconLogger
from reprlearn.callbacks.hist_logger import  HistogramLogger
from reprlearn.callbacks.beta_scheduler import BetaScheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# reprlearn helper functions
import reprlearn as rl
from reprlearn.utils.misc import info

from reprlearn.visualize.utils import show_timg, show_timgs, show_batch, make_grid_from_tensors
from reprlearn.utils.misc import info, now2str, today2str, get_next_version_path, n_iter_per_epoch
from reprlearn.utils.scheduler import frange_cycle_linear

# utils for instatiating a selected datamodule and a selected model
from utils import get_model_class, get_dm_class
from utils import instantiate_model, instantiate_dm
from utils import add_base_arguments

if __name__ == '__main__':

    empty_parser = argparse.ArgumentParser(add_help=False) # add_help=False is important!
    # ------------------------------------------------------------------------
    # Add general arguments for this CLI script for training/testing
    # ------------------------------------------------------------------------
    base_parser = add_base_arguments(empty_parser) # add or create a new parser obj
    base_args, unknown = base_parser.parse_known_args()
    print("Base CLI args: ")
    pprint(base_args)
    # breakpoint()
    # ------------------------------------------------------------------------
    # Add model/datamodule/trainer specific args
    # ------------------------------------------------------------------------
    model_class = get_model_class(base_args.model_name)
    dm_class = get_dm_class(base_args.data_name)

    model_parser = model_class.add_model_specific_args(empty_parser)
    dm_parser = dm_class.add_model_specific_args(empty_parser)
    trainer_parser = pl.Trainer.add_argparse_args(empty_parser)
    parents = [base_parser, model_parser, dm_parser, trainer_parser]

    # Create a parser with model, datamodule, trainer arguments
    parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')

    # ------------------------------------------------------------------------
    # Add Callback args
    # ------------------------------------------------------------------------
    # parser = BetaScheduler.add_argparse_args(parser)
    # parser.add_argument("--hist_epoch_interval", type=int, default=10,
    #                     help="Epoch interval to plot histogram of q's parameter"
    #                     )
    # parser.add_argument("--recon_epoch_interval", type=int, default=10,
    #                     help="Epoch interval to plot reconstructions of train and val samples"
    #                     )
    # Model Checkpoint Callback
    parser.add_argument('--ckpt_metric', type=str, default='val_tenenb  loss',
        help="Metric to decide (k) best model(s) to checkpoint. Default: val_loss"
    )
    parser.add_argument('--ckpt_mode', type=str, default='min',
        help="min or max; If min, lower ckpt_metric value means a better model. Default: min"
    )
    parser.add_argument('--save_top_k', type=int, default=5,
        help="Save top k best models during training; 0 saves None, -1 saves all; k>=2 as expected; See pytorch-lighting's doc"
    )
    # Early stopping Callback
    parser.add_argument('--stop_metric', type=str, default='val_loss',
        help="Metric to decide (k) best model(s) to checkpoint. Default: val_loss"
    )
    parser.add_argument('--stop_mode', type=str, default='min',
        help="min or max; If min, lower ckpt_metric value means a better model. Default: min"
    )
    parser.add_argument('--stop_patience', type=int, default='10',
        help="Stop training if the stop_metric does not get better over this number of epochs"
    )


    # ------------------------------------------------------------------------
    # Parse the args
    # ------------------------------------------------------------------------
    args = parser.parse_args()
    print("Final args: ")
    pprint(args)


    # ------------------------------------------------------------------------
    # Initialize model, datamodule, trainer using the parsered arg dict
    # ------------------------------------------------------------------------
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print("===GPUs===")
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    # Init datamodule and model
    dm = instantiate_dm(args)
    model = instantiate_model(args)

    # Init Tensorboard logger
    exp_name = f'{model.name}_{dm.name}'
    print('Exp name: ', exp_name)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_root,
                                             name=exp_name,
                                             default_hp_metric=False,
                                             )
    log_dir = Path(tb_logger.log_dir)
    print("Log Dir: ", log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print("Created: ", log_dir)

    # Init callbacks
    ckpt_callback = ModelCheckpoint(
        monitor=args.ckpt_metric,
        mode=args.ckpt_mode,
        save_top_k=args.save_top_k,
    )
    early_stopping_callback = EarlyStopping(
        monitor=args.stop_metric,
        patience=args.stop_patience,
        mode=args.stop_mode,
    )
    # callbacks = [ckpt_callback, early_stopping_callback]
    callbacks = [ckpt_callback] # todo: for GAN, model-ckpt logger and early_stopping cb should track FID (or some kind of eval metrics)

    # Init trainer
    trainer_overrides = {
        'gpus':1,
         'progress_bar_refresh_rate':0, # don't print out progress bar
        'terminate_on_nan':True,
        'check_val_every_n_epoch':10,
        'logger': tb_logger,
        'callbacks': callbacks
    }
    trainer = pl.Trainer.from_argparse_args(args, **trainer_overrides)

    # ------------------------------------------------------------------------
    # Run the experiment
    # ------------------------------------------------------------------------
    start_time = time.time()
    print(f"{exp_name} started...")
    print(f"Logging to {Path(tb_logger.log_dir).absolute()}")
    trainer.fit(model, dm)
    print(f"Finished at ep {trainer.current_epoch, trainer.batch_idx}")
    print(f"Training Done: took {time.time() - start_time}")

    # ------------------------------------------------------------------------
    # Log the best score and current experiment's hyperparameters
    # ------------------------------------------------------------------------
    hparams = model.hparams.copy()
    hparams.update(dm.hparams)
    best_score = trainer.checkpoint_callback.best_model_score.item()
    metrics = {'hparam/best_score': best_score}  # todo: define a metric and use it here
    trainer.logger.log_hyperparams(hparams, metrics)

    print("Logged hparams and metrics...")
    print("\t hparams: ")
    pprint(hparams)
    print("=====")
    print("\t metrics: ", metrics)
    print(f"Finished logging to Tensorboard")

    # ------------------------------------------------------------------------
    # TODO: Evaluation
    #   1. Reconstructions:
    #     x --> model.encoder(x) --> theta_z
    #     --> sample N latent codes from the Pr(z; theta_z)
    #     --> model.decoder(z) for each sampled z's
    #   2. Embedding:
    #       a mini-batch input -> mu_z, logvar_z
    #       -> rsample
    #       -> project to 2D -> visualize
    #   3. Inspect the topology/landscape of the learned latent space
    #     Latent traversal: Pick a dimension of the latent space.
    #     - Keep all other dimensions' values constant.
    #     - Vary the chosen dimenion's values (eg. linearly, spherically)
    #     - and decode the latent codes. Show the outputs of the decoder.
    #   4. Marginal Loglikelihood of train/val/test dataset
    # ------------------------------------------------------------------------
    print(f"Sample generation starting...")
    start_time = time.time()
    # model.eval()

    print(f"Sample generation done: took {time.time() - start_time}")

paPPP