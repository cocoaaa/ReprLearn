"""
Find the best hparam setting for a specific BiVAE model trained on a specific datamodule.
w or w/o the beta annealing
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

# Q: what would be a good balance btw kld_weight vs. adv_weight (w/o beta annealing)
# Run Hyperparameter-search
# - with kld_weight added to the search space
# - turn on/off kld_weight annealing (ie. BetaScheduler)

# Final Search space:
# is_contrasive, kld_weight, adv_loss_weight, learning_rate, batch_size

nohup python tune_loss_weights_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_ids 2 --max_epochs=300   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-16-ray/" &

Or, with MultiRotatedMNIST datamodule
nohup python tune_loss_weights_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--data_name="multi_rotated_mnist" --angles -45 0 45 --n_styles=3 \
--gpu_ids 1 --max_epochs=400   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23-ray/" &


Or, with MultiMaptiles Datamodule
nohup python tune_loss_weights_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities la paris \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=3 \
--zooms 14 \
--gpu_ids 1 2 --max_epochs=400   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23-ray/" &


# To see the hparam search progress:
# view the Ray dashboard at http://127.0.0.1:8265
# Run this at  local terminal:
# ssh -NfL 8265:localhost:8265 arya
"""

import os
import time
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import List, Set,Any, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import warnings
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# Ray
import ray
from ray import tune
from ray.tune import track

# from hyperopt import hp
# from ray.tune.suggest.hyperopt import HyperOptSearc

# callbacks
from src.callbacks.recon_logger import ReconLogger
from src.callbacks.hist_logger import  HistogramLogger
from src.callbacks.beta_scheduler import BetaScheduler
from pytorch_lightning.callbacks import LearningRateMonitor

# src helpers
from src.utils.misc import info, n_iter_per_epoch
from src.models.model_wrapper import ModelWrapper

# utils for instatiating a selected datamodule and a selected model
from .utils import get_model_class, get_dm_class
from .utils import instantiate_model, instantiate_dm


if __name__ == '__main__':

    parser = ArgumentParser()

    # Define general arguments for this CLI script for trianing/testing
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default='fit', help="fit or test")
    parser.add_argument("--log_root", type=str, default='./lightning_logs', help='root directory to save lightning logs')
    parser.add_argument("--gpu_ids", nargs='+', type=str,
                        help="GPU ID(s) to use") #Returns an empty list if not specified

    # Callback args
    # parser.add_argument("--hist_epoch_interval", type=int, default=10, help="Epoch interval to plot histogram of q's parameter")
    # parser.add_argument("--recon_epoch_interval", type=int, default=10, help="Epoch interval to plot reconstructions of train and val samples")

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
    # Callback switch args
    parser = BetaScheduler.add_argparse_args(parser)

    args, unknown = parser.parse_known_args()
    print("Final args: ")
    pprint(args)
    print("Final unknown :")
    pprint(unknown)
    # ------------------------------------------------------------------------
    # Initialize model, datamodule, trainer using the parsered arg dict
    # ------------------------------------------------------------------------
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu_ids)

    print("===GPUs===")
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # breakpoint()

    def set_hparam_and_train_closure(config: Dict[str, Any]) -> None:
        """
        Use the (k,v) in `config` to update the args
        :param config: Hyperparam search space as a Dict[hparam-name, value of the hpamram]. Eg:
            - latent_dim: int
            - is_contrasive: bool
            - batch_size: int
            - learning_rate: float
        :return: None. Train the model in the specified hyperparmeter space
        """
        print("Inside the clousure===")
        pprint(args)
        print("===")
        pprint(config)

        d_args =  vars(args)
        for k, v in config.items():
            d_args[k] = v
            print("Overwrote CLI args: ", k)

        # If kl_weight annealing is used,
        # the BetaScheduler's beta_step (ie. max_beta) will be set to the value of the kl_weight currently being searched
        if args.use_beta_scheduler:
            d_args['beta_stop'] = config['kld_weight']  # BetaSchduler's max_beta is kld_weight
            print("Setting the max_beta to kld_weight: ", d_args['beta_stop'])

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
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
            print("Created: ", log_dir)

        #Specify callbacks
        callbacks = [
            LearningRateMonitor(logging_interval='epoch')
            # HistogramLogger(hist_epoch_interval=args.hist_epoch_interval),
            # ReconLogger(recon_epoch_interval=args.recon_epoch_interval),
            #         EarlyStopping('val_loss', patience=10),
        ]
        if args.use_beta_scheduler:
            # n_epoch
            max_iters = n_iter_per_epoch(dm.train_dataloader()) * args.max_epochs
            callbacks.append(BetaScheduler(max_iters,
                                           start=args.beta_start,
                                           stop=args.beta_stop,
                                           n_cycle=args.beta_n_cycle,
                                           ratio=args.beta_ratio,
                                           log_tag=args.beta_log_tag))

        trainer_overwrites = {
            'gpus': 1, # use a single gpu
            'progress_bar_refresh_rate': 0, # don't print out progress bar
            'terminate_on_nan': True,
            'check_val_every_n_epoch': 10,
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
        # print("Evaluations...")
        # model.eval()

        # ------------------------------------------------------------------------
        # TODO: 1. Recon
        # ------------------------------------------------------------------------

        print(f"Done: took {time.time() - start_time}")

    # ------------------------------------------------------------------------
    # Specify hyperparameter search space
    # ------------------------------------------------------------------------
    search_space = {
        'is_contrasive': tune.grid_search([False, True]),

        'kld_weight': tune.grid_search([0., 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32., 64, 128., 256]),
        'use_beta_scheduler': tune.grid_search([True, False]),
        'adv_loss_weight': tune.grid_search([5., 15., 45., 135., 405., 1215.]),
        'learning_rate': tune.grid_search(list(np.logspace(-4., -1, num=10))),
        'batch_size': tune.grid_search([32, 64, 128, 256, 512, 1024]),
    }


    # ------------------------------------------------------------------------
    # Start hyperparameter search using Ray
    # ------------------------------------------------------------------------
    ray.shutdown()
    ray.init(log_to_driver=False)
    analysis = tune.run(
        set_hparam_and_train_closure,
        config=search_space,
        verbose=1,
        name="Tune-loss-weights-BiVAE", # logging directory
        resources_per_trial={"cpu":16, "gpu": len(args.gpu_ids)},
    )

    # dfs = analysis.fetch_trial_dataframes()





























