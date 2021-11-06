"""
trainer_main.py

Required args:
    --model_name: "vae" or "iwae"
    --data_name: "maptiles" or "mnist"
    --latent_dim: int, eg. 10

Optional args: (partial)
    --hidden_dims: eg. --hidden_dims 32 64 128 256 (which is default)

To run: (at the root of the project, ie. /data/hayley-old/Tenanbaum2000
nohup python src/train.py --model_name="vae" --data_name="mnist" --latent_dim=10
nohup python src/train.py --model_name="iwae" --data_name="mnist" --latent_dim=10

"""
# Load libraries
# In[2]:


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
import torch.nn.functional as F
import torch.optim as optim

import torchvision

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers

# plmodules
from reprlearn.models.plmodules.three_fcs import ThreeFCs
from reprlearn.models.plmodules.vanilla_vae import VanillaVAE
from reprlearn.models.plmodules.iwae import IWAE

# datamodules
from reprlearn.data.datamodules.maptiles_datamodule import MaptilesDataModule
from reprlearn.data.datamodules.mnist_datamodule import MNISTDataModule

# callbacks
from reprlearn.callbacks.recon_logger import ReconLogger
from reprlearn.callbacks.hist_logger import  HistogramLogger

# src helpers
from reprlearn.utils.misc import info
from reprlearn.models.model_wrapper import ModelWrapper


def get_act_fn(fn_name:str) -> Callable:
    fn_name = fn_name.lower()
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
    }[fn_name]

def get_model_class(model_name: str) -> object:
    model_name = model_name.lower()
    return {
        "three_fcs": ThreeFCs,
        "vae": VanillaVAE,
        "iwae": IWAE,
    }[model_name]


def get_dm_class(dm_name:str) -> object:
    dm_name = dm_name.lower()
    return {
        'mnist': MNISTDataModule,
        'maptiles': MaptilesDataModule
    }[dm_name]

def instantiate_dm(args):
    data_name = args.data_name.lower()
    data_root = Path(args.data_root)

    if data_name == 'mnist':
        kwargs = {
            'data_root': data_root,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'verbose': args.verbose,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
        }
        dm = MNISTDataModule(**kwargs)

    elif data_name == 'maptiles':
        kwargs = {
            'data_root': data_root,
            'cities': args.cities,
            'styles': args.styles,
            'zooms': args.zooms,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'verbose': args.verbose,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
        }
        dm = MaptilesDataModule(**kwargs)
    else:
        raise KeyError("Data name must be in ['mnist', 'maptiles']")

    return dm


def instantiate_model(args):
    act_fn = get_act_fn(args.act_fn)

    kwargs = {
        'in_shape': args.in_shape, #dm.size()
        'latent_dim': args.latent_dim,
        'hidden_dims': args.hidden_dims,
        'act_fn': act_fn,
        'learning_rate': args.learning_rate,
        'verbose': args.verbose,
    }
    model_name = args.model_name
    model_class = get_model_class(model_name)

    if model_name == 'iwae':
        kwargs['n_samples'] = args.n_samples

    return model_class(**kwargs)


if __name__ == '__main__':

    parser = ArgumentParser()

    # Define general arguments for this CLI script for trianing/testing
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default='fit', help="fit or test")
    parser.add_argument("--log_root", type=str, default='./lightning_logs', help='root directory to save lightning logs')
    parser.add_argument("--gpu_id", type=str, required=True, help="ID of GPU to use")

    # Callback args
    parser.add_argument("--hist_epoch_interval", type=int, default=10, help="Epoch interval to plot histogram of q's parameter")
    parser.add_argument("--recon_epoch_interval", type=int, default=10, help="Epoch interval to plot reconstructions of train and val samples")

    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    print("CLI args: ")
    pprint(args)

    # ------------------------------------------------------------------------
    # Add model/datamodule/trainer specific args
    # ------------------------------------------------------------------------
    model_class = get_model_class(args.model_name)
    dm_class = get_dm_class(args.data_name)
    parser = model_class.add_model_specific_args(parser)
    parser = dm_class.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    print("Final args: ")
    pprint(args)
    # ------------------------------------------------------------------------
    # Initialize model, datamodule, trainer using the parsered arg dict
    # ------------------------------------------------------------------------
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Init. datamodule and model
    dm = instantiate_dm(args)
    model = instantiate_model(args)

    # Specify logger and callbacks
    exp_name = f'{model.name}_{dm.name}'
    print('Exp name: ', exp_name)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_root,
                                             name=exp_name,
                                             default_hp_metric=False,
                                             )
    callbacks = [
        HistogramLogger(hist_epoch_interval=args.hist_epoch_interval),
        ReconLogger(recon_epoch_interval=args.recon_epoch_interval),
        #         EarlyStopping('val_loss', patience=10),
    ]

    overwrites = {
        'gpus':1,
         'progress_bar_refresh_rate':20,
        'terminate_on_nan':True,
        'check_val_every_n_epoch':10,
        'logger': tb_logger,
        'callbacks': callbacks
    }

    # Init. trainer
    trainer = pl.Trainer.from_argparse_args(args, **overwrites)

    # Log model's computational graph
    model_wrapper = ModelWrapper(model)
    # tb_logger.experiment.add_graph(model_wrapper, model.)
    tb_logger.log_graph(model_wrapper)


    # ------------------------------------------------------------------------
    # Run the experiment
    # ------------------------------------------------------------------------
    start_time = time.time()
    print(f"{exp_name} started... Logging to {Path(tb_logger.log_dir).absolute()}")
    trainer.fit(model, dm)
    print(f"Finished at ep {trainer.current_epoch, trainer.batch_idx}")


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

    # ------------------------------------------------------------------------
    # Evaluation
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
    print("Evaluations...")
    model.eval()

    # ------------------------------------------------------------------------
    # 1. Recon
    # ------------------------------------------------------------------------


    print(f"Done: took {time.time() - start_time}")




