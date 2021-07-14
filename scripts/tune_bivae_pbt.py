"""
Find the best hparam setting for a specific BiVAE model trained on a specific datamodule.

Required args:
    --model_name: eg. "vae", "iwae", "bivae"
    --data_name: eg. "maptiles", "mnist", "multi_mono_mnist"
    --latent_dim: int, eg. 10

Optional args: (partial)
    --hidden_dims: eg. --hidden_dims 32 64 128 256 (which is default)

Hyperparameter space:
- latent_dim = [16, 32, 63, 128]
- is_contrasive =  [False, True]
- kld_weight = [
- adv_loss_weight = [5, 15, 45, 135, 405, 1215]
- batch_size = [32, 64, 128, 256, 514, 1028]
- learning_rate =

To run: (at the root of the project, ie. /data/hayley-old/Tenanbaum2000
# Values for adv_weight, latent_dim, batch_size, lr, is_contrasive will be overwritten
# as the searched hyperparmeter values

 nohup python tune_hparams_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=300 --batch_size=128 -lr 1e-3  --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-13-ray/" &

 nohup python tune_hparams_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
--use_beta_scheduler \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=300 --batch_size=128 -lr 1e-3  --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-14-ray/" &

# View the Ray dashboard at http://127.0.0.1:8265
# Run this at  local terminal:
# ssh -NfL 8265:localhost:8265 arya

"""

import os
import time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path
from typing import List, Set,Any, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import warnings
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn

import torchvision
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.cloud_io import load as pl_load

# Ray
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from src.callbacks.recon_logger import ReconLogger
from src.callbacks.hist_logger import  HistogramLogger
from src.callbacks.beta_scheduler import BetaScheduler

# src helpers
from src.utils.misc import info, n_iter_per_epoch
from src.models.model_wrapper import ModelWrapper

# utils for instatiating a selected datamodule and a selected model
from utils import get_model_class, get_dm_class
from utils import instantiate_model, instantiate_dm
from utils import add_base_arguments

def train_tune_checkpoint(args: Union[Dict, Namespace],
               checkpoint_dir=None,
               checkpoint_fn='checkpoint'):
    # Init. datamodule and model
    dm = instantiate_dm(args)
    dm.setup('fit')

    # Init model
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
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print("Created: ", log_dir)

    # Specify callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        TuneReportCheckpointCallback(
            metrics={
            'loss': 'val_loss',
            'mean_accuracy': 'val/style_acc', # use the string after pl.Module's "self.log("
            },
            filename=checkpoint_fn, # where is this checkpoint_dir exactly?
            on="validation_end"
        ),
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
        # 'logger' = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."), #for Ray PBT scheduler
        'callbacks': callbacks,
    }

    # Init. trainer
    trainer = pl.Trainer.from_argparse_args(args, **trainer_overwrites)

    # Optionally, load saved state_dict from the checkpoint
    if checkpoint_dir is not None:
        ckpt = pl_load(os.path.join(checkpoint_dir, checkpoint_fn),
                       map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['state_dict'])
        trainer.current_epoch = ckpt["epoch"]

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
    # hparams = model.hparams.copy()
    # hparams.update(dm.hparams)
    # best_score = trainer.checkpoint_callback.best_model_score.item()
    # metrics = {'hparam/best_score': best_score}  # todo: define a metric and use it here
    # trainer.logger.log_hyperparams(hparams, metrics)
    #
    # print("Logged hparams and metrics...")
    # print("\t hparams: ")
    # pprint(hparams)
    # print("=====")
    # print("\t metrics: ", metrics)
    # print(f"Training Done: took {time.time() - start_time}")


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

    # RayTune args
    parser.add_argument('--n_cpus',  type=int, default=8, help='Num of CPUs per trial')
    parser.add_argument("--gpu_ids", type=str, required=True, nargs='*',
                        help="GPU ID(s) to use") #Returns an empty list if not specified
    parser.add_argument("--n_ray_samples", type=int, default=1,
                         help="Num of Ray Tune's run argument, num_samples")
    parser.add_argument("--ray_log_dir", type=str, default="/data/log/ray/pbt",
                        help="dir to save training results from Ray")
    # Callback switch args
    parser = BetaScheduler.add_argparse_args(parser)
    # parser.add_argument("--hist_epoch_interval", type=int, default=10, help="Epoch interval to plot histogram of q's parameter")
    # parser.add_argument("--recon_epoch_interval", type=int, default=10, help="Epoch interval to plot reconstructions of train and val samples")

    args = parser.parse_args()
    print("Final args: ")
    pprint(args)

    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu_ids)
    print("===GPUs===")
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    def set_hparam_and_train_closure(config: Dict[str, Any]):
        """Use the (k,v) in `overwrite` to update the args

        Parameters
        ----------
        config: Hyperparam search space as a Dict[hparam-name, value of the hpamram]
            This dict object is a sample point from the Ray's Hyperparameter space,
            and will be used to overwrite the `args`'s key-value with its key-value.

        Returns
        -------
        None. Train the model in the specified hyperparmeter space
        """
        print("Inside the clousure===")
        pprint(args)
        print("===")
        pprint(config)

        d_args =  vars(args)
        for k, v in config.items():
            d_args[k] = v
            print("Overwrote args: ", k)

        # Start experiment with this overwritten hyperparams
        train_tune_checkpoint(args)

    # ------------------------------------------------------------------------
    # Specify hyperparameter search space
    # ------------------------------------------------------------------------
    # search_space = {
    #     "latent_dim": 20, #tune.grid_search([16, 32, 64,128]),
    #     'is_contrasive': tune.grid_search([False, True]),
    #     'adv_loss_weight': tune.grid_search([5., 15., 45., 135., 405., 1215.]),
    #     'learning_rate': tune.grid_search(list(np.logspace(-4., -1, num=10))),
    #     'batch_size': tune.grid_search([32, 64, 128, 256, 512, 1024]),
    # }
    search_space = {
        # "latent_dim": tune.grid_search([10, 20, 60, 100]),
        # 'enc_type': tune.grid_search(['conv', 'resnet']),
        # 'dec_type': tune.grid_search(['conv', 'resnet']),

        'is_contrasive': tune.grid_search([False, True]),
        'kld_weight': tune.grid_search([0.5, 1.0, 4.0, 16.0, 64, 256, 1024]), # can be tuned via PBT WHILE training
        'use_beta_scheduler': False, #tune.grid_search([False,True]),  # can be tuned via PBT WHILE training
        'adv_loss_weight': tune.grid_search([5., 15., 45., 135., 405., 1215.]),  # can be tuned via PBT WHILE training

        'learning_rate': 1e-3, #tune.grid_search(list(np.logspace(-4., -1, num=10))),
        'batch_size': 32, #tune.grid_search([32, 64, 128,]),
    }

    # ------------------------------------------------------------------------
    # Start hyperparameter search using Ray
    # ------------------------------------------------------------------------
    ray.shutdown()
    ray.init(log_to_driver=False)
    # search_alg
    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": [32, 64, 128]
        })
    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        set_hparam_and_train_closure,
        config=search_space,
        metric='loss',
        mode='min',
        num_samples=args.n_ray_samples,
        verbose=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="Tune-BiVAE-PBT", # name of experiment
        local_dir= args.ray_log_dir,
        resources_per_trial={"cpu":args.n_cpus, "gpu": len(args.gpu_ids)}, # there are 16cpus in arya machine; so at a time 16/2=8 trials will be run concurrently
    )
    print("Best hyperparameters found were: ", analysis.best_config)

    # dfs = analysis.fetch_trial_dataframes()





























