#!/usr/bin/env python
# coding: utf-8
# Usage
# -----
# python test_gan_on_datamodules.py --data_name mnist --gpu_id 0
# python test_gan_on_datamodules.py --data_name mnistm --gpu_id 0
# python test_gan_on_datamodules.py --data_name mono_mnist --gpu_id 2
# python test_gan_on_datamodules.py --data_name usps --gpu_id 1
#

import os
import time
from pathlib import Path
import logging
import argparse
from argparse import Namespace
import torch
import pytorch_lightning as pl
# DataModules
from reprlearn.data.datamodules import MNISTDataModule
from reprlearn.data.datamodules import USPSDataModule
from reprlearn.data.datamodules import MNISTMDataModule
from reprlearn.data.datamodules import MonoMNISTDataModule
# PLModules
from reprlearn.models.conv_generator import ConvGenerator
from reprlearn.models.fc_discriminator import FCDiscriminator
from reprlearn.models.plmodules.gan import GAN
# Callbacks
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# Utils
from reprlearn.utils.misc import n_iter_per_epoch, today2str

# Set Path to data root dirs
# - DATA_ROOT:
#   - Use '/data/hayley-old/Tenanbaum2000/data' for MNIST, Mono-MNIST, Rotated-MNIST, Teapots
#   - Use `/data/hayley-old/maptiles_v2' folder for Maptile dataset
ROOT = Path("/data/hayley-old/Tenanbaum2000")
DATA_ROOT = Path("/data/hayley-old/Tenanbaum2000/data")
# DATA_ROOT = Path("/data/hayley-old/maptiles_v2/")

# MNIST datamodule
mnist_dm = MNISTDataModule(
    data_root=DATA_ROOT,
    in_shape=(1,32,32),
    batch_size=32,
    download=True,
)

# MNIST-M datamodule
mnistm_dm = MNISTMDataModule(data_root=DATA_ROOT/'MNIST-M',
                       in_shape=(3, 32,32),
                      batch_size=32)

# Mono-mnist datamodule
color ='blue'
seed = 321
mono_mnist_dm = MonoMNISTDataModule(
    data_root=DATA_ROOT/'Mono-MNIST',
    color=color,
    seed=seed,
    in_shape=(3,32,32),
    batch_size=32
)

# USPS datamodule
usps_dm = USPSDataModule(data_root=DATA_ROOT/'USPS',
                       in_shape=(1,32,32),
                      batch_size=32)

dms = {
    'mnist': mnist_dm,
    'mnistm': mnistm_dm,
    'mono_mnist': mono_mnist_dm,
    'usps': usps_dm
}
for dm in dms.values():
    dm.setup('fit')

# Test if each datamodule works with each plmodule obj
# Check a simple vae and a simple gan module's def training_step to use its datamodule's unpack
def train_gan_on_dm(args: Namespace):
    # extract experiment parameters
    data_name = args.data_name
    max_epochs = args.max_epochs
    early_stop = args.early_stop
    start = time.time()
    # Init LightningModule
    # Generator
    dm = dms[data_name]
    in_shape = dm.in_shape
    latent_dim = 10
    len_flatten = 32 #latent_dim * 4
    decoder_dims = [256, 128, 64, 32]
    dec_type = 'conv'
    # dec_type = 'resnet'
    gen = ConvGenerator(latent_dim=latent_dim,
                        latent_emb_dim=len_flatten,
                        dec_hidden_dims=decoder_dims,
                        in_shape=in_shape,
                        dec_type='conv'
                       )
    # Discriminator
    discr = FCDiscriminator(in_shape=in_shape)

    #GAN
    lr_g = 1e-3
    lr_d = 1e-3
    niter_D_per_G = 5
    b1 = 0.99
    b2 = 0.99
    # logging params
    log_every = 5 # log sample images every 5 eps
    n_show = 64 # show 64 generated images

    model = GAN(
        in_shape=in_shape,
        latent_dim=latent_dim,
        generator=gen,
        discriminator=discr,
        lr_g=lr_g,
        lr_d=lr_d,
        b1=b1,
        b2=b2,
        niter_D_per_G=niter_D_per_G,
    )

    # ## Init pl.Trainer
    # Init. callbacks
    # Model Checkpoint criterion
    ckpt_metric, ckpt_mode = 'val/loss_G', 'min'
    # ckpt_metric, ckpt_mode = 'loss', 'min'
    ckpt_callback = ModelCheckpoint(
        monitor=ckpt_metric,
        mode=ckpt_mode,
        save_top_k=5,
    )

    stop_metric, stop_mode = 'val/loss_G', 'min'
    # stop_metric, stop_mode = 'loss', 'min'
    stop_patience = 100
    early_stopping_callback = EarlyStopping(
        monitor=stop_metric,
        patience=stop_patience,
        mode=stop_mode,
    )

    callbacks = [ckpt_callback]
    if early_stop:
        callbacks.append(early_stopping_callback)

    # Init. Tensorboard logger
    exp_name = f'{model.name}_{dm.name}'
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=f'{ROOT}/lightning_logs/test/{today2str()}',
        name=exp_name,
        default_hp_metric=False, #todo: what is this param's effect?
    )

    log_dir = Path(tb_logger.log_dir) # lightning_logs/test/<DATE>/<model-name>-<dm-name>/version_X
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
    logging.info(f'Dataset: {data_name}')
    logging.info(f"Model: {model.name}")
    logging.info(f'GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    logging.info(f"Log Dir:  {log_dir.absolute()}")

    # Init. pl.Trainer
    trainer_config = {
        'gpus':1,
        'max_epochs': max_epochs,
        'progress_bar_refresh_rate':0,
        'terminate_on_nan':True,
        'check_val_every_n_epoch': 1,
        'logger':tb_logger,
        'callbacks':callbacks,
    }
    trainer = pl.Trainer(**trainer_config) # for real training

    # Fit model
    trainer.fit(model, dm)
    logging.info(f"Finished training at ep {trainer.current_epoch}")
    logging.info(f"\tTook {(time.time() - start)/60} mins")

    # ------------------------------------------------------------------------
    # Log the best score and current experiment's hyperparameters
    # ------------------------------------------------------------------------
    hparams = model.hparams.copy()
    hparams.update(dm.hparams)
    hparams['ckpt_metric'] = ckpt_metric
    if early_stop:
            hparams['stop_metric'] = stop_metric

    try:
        best_ckpt_score = trainer.checkpoint_callback.best_model_score.item()
    except:
        best_ckpt_score = -1
    metrics = {
        f'hparam/best_{ckpt_metric}': best_ckpt_score
    }

    trainer.logger.log_hyperparams(hparams, metrics)
    logging.info("Logged hparams and metrics...")
    logging.info(f"\thparams: {hparams}")
    logging.info(f"\tmetrics: {metrics}")
    logging.info("Finished logging to Tensorboard")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True,
                        help="One of mnist, mnistm, mono_mnist, usps, maptiles")
    parser.add_argument('--gpu_id', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)

    parser.add_argument('--early_stop', dest='early_stop', action='store_true')
    parser.add_argument('--no_early_stop', dest='early_stop', action='store_false')
    parser.set_defaults(early_stop=True)

    args = parser.parse_args()

    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # run test
    train_gan_on_dm(args)

