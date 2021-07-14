"""train.py
Train a single configuration of a model specified on a specified data

Required CLI args
-----------------
    --model_name: "vae", "beta_vae", "iwae", "bivae"
    --data_name: "mnist", "multi_rotated_mnist", "multi_mono_mnist",
                "maptiles", "multi_maptiles"
    --latent_dim: int, eg. 10

Optional CLI args: (partial)
----------------------------
    --hidden_dims: eg. --hidden_dims 32 64 128 256 (which is default)

Note
----
  Each model (specified by --model_name) and datamodule (specified by --data_name)
expects a different set of arguments. For example, `bivae` models allow the following
arguments:
Required:
--n_styles
--adversary_dim
--adv_loss_weight


To run: (at the root of the project, ie. /data/hayley-old/Tenanbaum2000
nohup python train.py --model_name="vae" --data_name="mnist" --latent_dim=10
nohup python train.py --model_name="iwae" --data_name="mnist" --latent_dim=10


# Train BetaVAE on MNIST
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &


# Train BiVAE on Multi Monochrome MNIST
nohup python train.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64  --adv_dim 32 32 -lr 1e-3 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1


# Train BiVAE on Multi Rotated MNIST
nohup python train.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--data_name="multi_rotated_mnist" --angles -45 0 45 --n_styles=3 \
--gpu_id=1

nohup python train.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 64 64 --adv_dim 32  \
--data_name="multi_rotated_mnist" --angles 0 15 30 45 60 --n_styles=5 \
--gpu_id=3 --max_epochs=400   --terminate_on_nan=True  \
-lr 3e-3 --adv_weight 2000.0 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-05-23/" &

## Specify which indices to use among the MNIST -- comparable to DIVA's experiments
## change 0 to anything inbtw 0,...,9
nohup python train.py --model_name="bivae" \
--latent_dim=128 --hidden_dims 32 64 64 64 --adv_dim 32 32 32 \
--data_name="multi_rotated_mnist" --angles -45 0 45 --n_styles=3 \
--selected_inds_fp='/data/hayley-old/Tenanbaum2000/data/Rotated-MNIST/supervised_inds_0.npy' \
--gpu_id=1


# Train BiVAE on Multi Maptiles MNIST
nohup python train.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
--data_name="multi_maptiles" \
--cities la paris \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=3 \
--zooms 14 \
--gpu_id=2 --max_epochs=400   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &


# Train BiVAE on Multi OSMnxRoads
nohup python train.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 1.0 \
--data_root="/data/hayley-old/osmnx_data/images" \
--data_name="osmnx_roads" \
--cities paris \
--bgcolors "k" "r" "g" "b" "y" --n_styles=5 \
--zooms 14 \
--gpu_id=2 --max_epochs=300   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-05-18/" &

nohup python train.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 1.0 \
--data_root="/data/hayley-old/osmnx_data/images" \
--data_name="osmnx_roads" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
     'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' \
     'berlin' 'montreal' 'rome' \
--bgcolors "k" "r" "g" "b" "y" --n_styles=5 \
--zooms 14 \
--gpu_id=2 --max_epochs=300   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-05-18/" &


"""
import os,sys
from datetime import datetime
import time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import warnings
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


# callbacks
from src.callbacks.recon_logger import ReconLogger
from src.callbacks.hist_logger import  HistogramLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from src.callbacks.beta_scheduler import BetaScheduler

# src helpers
from src.utils.misc import info
from src.models.model_wrapper import ModelWrapper
from src.utils.misc import info, n_iter_per_epoch

# utils for instatiating a selected datamodule and a selected model
from utils import get_model_class, get_dm_class
from utils import instantiate_model, instantiate_dm
from utils import add_base_arguments

def train(args: Union[Dict, Namespace]):
    # Init. datamodule and model
    dm = instantiate_dm(args)
    dm.setup('fit')
    model = instantiate_model(args)

    # Specify logger
    exp_name = f'{model.name}_{dm.name}'
    print('Exp name: ', exp_name)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_root,
                                             name=exp_name,
                                             default_hp_metric=False,
                                             )
    log_dir = Path(tb_logger.log_dir)
    print("Log Dir: ", log_dir)
    # breakpoint()
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print("Created: ", log_dir)

    # Specify callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch')
        # HistogramLogger(hist_epoch_interval=args.hist_epoch_interval),
        # ReconLogger(recon_epoch_interval=args.recon_epoch_interval),
        #         EarlyStopping('val_loss', patience=10),
    ]
    if args.use_beta_scheduler:
        max_iters = n_iter_per_epoch(dm.train_dataloader()) * args.max_epochs
        callbacks.append(BetaScheduler(max_iters,
                                       start=args.beta_start,
                                       stop=args.beta_stop,
                                       n_cycle=args.beta_n_cycle,
                                       ratio=args.beta_ratio,
                                       log_tag=args.beta_log_tag))

    trainer_overwrites = {
        'gpus':1, #use a single gpu
        'progress_bar_refresh_rate':0, # don't print out progress bar
        'terminate_on_nan':True,
        'check_val_every_n_epoch':10,
        'logger': tb_logger,
        'callbacks': callbacks
    }

    # Init. trainer
    trainer = pl.Trainer.from_argparse_args(args, **trainer_overwrites)

    # Log model's computational graph
    model_wrapper = ModelWrapper(model)
    # tb_logger.experiment.add_graph(model_wrapper, model.)
    tb_logger.log_graph(model_wrapper)


    # ------------------------------------------------------------------------
    # Run the experiment
    # ------------------------------------------------------------------------
    start_time = time.time()
    print(f"{exp_name} started...")
    print(f"Logging to {Path(tb_logger.log_dir).absolute()}")
    trainer.fit(model, dm)
    print(f"Finished at ep {trainer.current_epoch, trainer.batch_idx}")


    # ------------------------------------------------------------------------
    # Log the best score and current experiment's hyperparameters
    # ------------------------------------------------------------------------
    hparams = model.hparams.copy()
    hparams.update(dm.hparams)
    hparams['use_beta_scheduler'] = args.use_beta_scheduler

    best_score = trainer.checkpoint_callback.best_model_score.item()
    metrics = {'hparam/best_score': best_score}  # todo: define a metric and use it here
    trainer.logger.log_hyperparams(hparams, metrics)

    print("Logged hparams and metrics...")
    print("\t hparams: ")
    pprint(hparams)
    print("=====")
    print("\t metrics: ", metrics)
    print(f"Training Done: took {time.time() - start_time}")

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
    # print("Evaluations...")
    # model.eval()


if __name__ == '__main__':
    parser = ArgumentParser()

    # ------------------------------------------------------------------------
    # Add general arguments for this CLI script for training/testing
    # ------------------------------------------------------------------------
    parser = add_base_arguments(parser)
    args, unknown = parser.parse_known_args()
    print("Base CLI args: ")
    pprint(args)

    # ------------------------------------------------------------------------
    # Add model/datamodule/trainer specific args
    # ------------------------------------------------------------------------
    model_class = get_model_class(args.model_name)
    dm_class = get_dm_class(args.data_name)
    parser = model_class.add_model_specific_args(parser)
    parser = dm_class.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--gpu_id", type=str, required=True, help="ID of GPU to use")

    # Callback switch args
    parser = BetaScheduler.add_argparse_args(parser)
    # parser.add_argument("--hist_epoch_interval", type=int, default=10, help="Epoch interval to plot histogram of q's parameter")
    # parser.add_argument("--recon_epoch_interval", type=int, default=10, help="Epoch interval to plot reconstructions of train and val samples")

    args = parser.parse_args()
    print("Final args: ")
    pprint(args)
    # breakpoint()
    # ------------------------------------------------------------------------
    # Run the training workflow
    # -- Select Visible GPU
    # -- Initialize model, datamodule, trainer using the parsed arg dict.
    # -- Specify callbacks
    # -- Init. trainer
    # -- Run the experiment
    # -- Log the best score and current experiment's hyperparameters
    # -- TODO: Add Evaluation
    # ------------------------------------------------------------------------
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Start experiment
    train(args)




