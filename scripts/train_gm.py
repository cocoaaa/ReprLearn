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
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import logging
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torchvision
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
from reprlearn.callbacks import DebugCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


# reprlearn helper functions
import reprlearn as rl
from reprlearn.utils.misc import info
from reprlearn.visualize.utils import show_timg, show_timgs, show_batch, make_grid_from_tensors
from reprlearn.utils.misc import info, now2str, today2str, get_next_version_path, n_iter_per_epoch

# utils for instatiating a selected datamodule and a selected model
from utils import get_model_class, get_dm_class
from utils import instantiate_model, instantiate_dm
from utils import add_base_arguments
from utils import parse_step_idx

# make ckpts and save sample set
from reprlearn.models.plmodules.utils import get_best_ckpt_path, get_best_k_ckpt_paths
from reprlearn.models.plmodules.utils import load_model_ckpt, load_best_model
from reprlearn.models.plmodules.utils import save_sample_from_model_ckpt
from reprlearn.utils.misc import change_suffix

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
    parser.add_argument('--ckpt_metric', type=str, default='val_loss',
        help="Metric to decide (k) best model(s) to checkpoint. Default: val_loss"
    )
    parser.add_argument('--ckpt_mode', type=str, default='min',
        help="min or max; If min, lower ckpt_metric value means a better model. Default: min"
    )
    parser.add_argument('--save_top_k', type=int, default=5,
        help="Save top k best models during training; 0 saves None, -1 saves all; k>=2 as expected; See pytorch-lighting's doc"
    )
    # Early stopping Callback
    parser.add_argument('--early_stop', dest='early_stop', action='store_true',
                        help="Use --early_stop to register early-stop callback")
    parser.add_argument('--no_early_stop', dest='early_stop', action='store_false',
                        help="Use --no_early_stop to exclude early-stop callback")
    parser.set_defaults(early_stop=True)

    parser.add_argument('--stop_metric', type=str, default='val_loss',
        help="Metric to decide (k) best model(s) to checkpoint. Default: val_loss"
    )
    parser.add_argument('--stop_mode', type=str, default='min',
        help="min or max; If min, lower ckpt_metric value means a better model. Default: min"
    )
    parser.add_argument('--stop_patience', type=int, default='10',
        help="Stop training if the stop_metric does not get better over this number of epochs"
    )

    # todo: Add image sample recon callback
    parser.add_argument('--write_to_disk_every', type=int, default='1',
        help="Interval to write sample from current generator (epoch)"
    )

    # ------------------------------------------------------------------------
    # Add sample generation args (after training)
    # ------------------------------------------------------------------------
    parser.add_argument('--num_generated_sample', type=int, default=10000,
                        help='number of datapts to generate from a trained model')

    # flags to save or not-save generated samples
    parser.add_argument('--save_sample', dest='save_sample', action='store_true',
                        help="Use --save_sample to save generated sample to disk (.pkl)")
    parser.add_argument('--no_save_sample', dest='save_sample', action='store_false',
                        help="Use --no_save_sample to not save generated sample to disk")
    parser.set_defaults(save_sample=True)

    # flags to save snapshot of generated samples (as png)
    parser.add_argument('--save_sample_snapshot', dest='save_sample_snapshot', action='store_true',
                        help="Use --save_sample_snapshot to save random samples of generated datapts (.png)")
    parser.add_argument('--no_save_sample_snapshot', dest='save_sample_snapshot', action='store_false',
                        help="Use --no_save_sample_snapshot to not save snapshot of generated datapts to disk")
    parser.set_defaults(save_sample_snapshot=True)

    # flags to save or not-save generated samples
    parser.add_argument('--log_sample_to_tb', dest='log_sample_to_tb', action='store_true',
                        help="Use --log_sample_to_tb to save generated sample to disk (.pkl)")
    parser.add_argument('--no_log_sample', dest='log_sample', action='store_false',
                        help="Use --no_log_sample_to_tb to not save generated sample to disk")
    parser.set_defaults(log_sample=True)

    # ------------------------------------------------------------------------
    # Parse the args (final args)
    # ------------------------------------------------------------------------
    args = parser.parse_args()
    print("Final args: ")
    # pprint(args)
    # breakpoint()

    # ------------------------------------------------------------------------
    # Initialize model, datamodule, trainer using the parsered arg dict
    # ------------------------------------------------------------------------
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # print("===GPUs===")
    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    # Init datamodule and model
    dm = instantiate_dm(args)
    model = instantiate_model(args)

    # Init Tensorboard logger
    exp_name = f'{model.name}_{dm.name}'
    # print('Exp name: ', exp_name)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_root,
                                             name=exp_name,
                                             default_hp_metric=False,
                                             )
    log_dir = Path(tb_logger.log_dir) # lightning_logs/<DATE>/<model-name>-<dm-name>/version_X
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print("Created log_dir:", log_dir.absolute())

    # Set logger to log messages to a local file "train.log"
    log_fp = log_dir / "train.log"
    logging.basicConfig(filename=log_fp,
                        filemode='w',
                        format='%(asctime)s -- %(message)s',
                        datefmt="%m/%d/%Y %I:%M%S %p",
                        level=logging.DEBUG)
    
    # Log initial experiment setup configs
    logging.info(f'New experiment started')
    logging.info(f'Exp name:  {exp_name}')
    logging.info(f"Model: {model.name}")
    logging.info(f'GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    logging.info(f'Final args:  {args}')
    logging.info(f"Log Dir:  {log_dir.absolute()}")


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
    sample_dir = log_dir/"Samples"
    if not sample_dir.exists():
        sample_dir.mkdir(parents=True)
        logging.info(f"Sample dir: {sample_dir.absolute()} -- created.")
    disk_logger = DebugCallback(
        log_every = args.write_to_disk_every,
        num_samples = 64,
        save_dir=sample_dir
    )
    callbacks = [ckpt_callback, disk_logger]
    if args.early_stop:
        callbacks.append(early_stopping_callback)
    # todo: add other callback in a similar way

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
    trainer.fit(model, dm)
    logging.info(f"Finished at ep {trainer.current_epoch}")
    logging.info(f"Training Done: took {(time.time() - start_time)/60} mins")

    # ------------------------------------------------------------------------
    # Log the best score and current experiment's hyperparameters
    # ------------------------------------------------------------------------
    hparams = model.hparams.copy()
    hparams.update(dm.hparams)
    # hparams['use_beta_scheduler'] = use_beta_scheduler
    hparams['ckpt_metric'] = args.ckpt_metric
    if args.early_stop:
        hparams['stop_metric'] = args.stop_metric
        
    try:
        best_ckpt_score = trainer.checkpoint_callback.best_model_score.item()
    except:
        best_ckpt_score = -1
    metrics = {
        f'hparam/best_{args.ckpt_metric}': best_ckpt_score
    }

    trainer.logger.log_hyperparams(hparams, metrics)
    logging.info("Logged hparams and metrics...")
    logging.info(f"\thparams: {hparams}")
    logging.info(f"\tmetrics: {metrics}")
    logging.info("Finished logging to Tensorboard")

    # ------------------------------------------------------------------------
    # Generate samples and save to disk
    # - need it for evaluation
    # TODO: Evaluation of VAE
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
    logging.info(f"\nSample generation starts...")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    sample_dir = Path(trainer.log_dir) / 'samples'
    logging.info(f'Sample dir: {sample_dir}')
    if not sample_dir.exists():
        sample_dir.mkdir()
        logging.info('\tCreated.')

    # Generate sample set from each ckpt
    num_generated_sample = args.num_generated_sample
    save_sample = args.save_sample
    save_sample_snapshot = args.save_sample_snapshot
    log_sample_to_tb = args.log_sample_to_tb
    n_show = args.n_show
    k_ckpts = get_best_k_ckpt_paths(trainer)
    for ckpt in k_ckpts:
        load_model_ckpt(model, ckpt)

        model.to(device)
        model.eval()
        sample = model.sample(num_generated_sample, device)

        # Save to disk
        sample_fp = sample_dir / f'{dm.name}-{Path(ckpt).stem}.pkl'

        if save_sample:
            torch.save(sample, sample_fp)
            logging.info(f'Saved {num_generated_sample} samples: ', sample_fp)

        if save_sample_snapshot:
            png_fp = change_suffix(sample_fp, '.png')
            inds = np.random.randint(0, num_generated_sample, size=n_show)

            f, _ = show_timgs(sample[inds].cpu(),
                              cmap='gray' if args.in_shape[0] == 1 else None,
                              title=sample_fp.stem)
            f.tight_layout()
            f.savefig(png_fp)
            logging.info('Saved snapshot of samples: ', png_fp)

        # log subset of generated samples to Tensorboard
        if log_sample_to_tb:
            ckpt_step = parse_step_idx(ckpt)
            inds = np.random.randint(0, num_generated_sample, size=n_show)
            grid = torchvision.utils.make_grid(sample[inds])
            trainer.logger.experiment.add_image("generated_images", grid, ckpt_step)

    logging.info(f"Sample generation done: took {(time.time() - start_time/60)} mins")





