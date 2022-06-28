#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import logging
import pprint
# pytorch, pl
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
# ReprLearn lib
from reprlearn.visualize.utils import show_timg, show_timgs, show_batch, make_grid_from_tensors
from reprlearn.utils.misc import info, now2str, today2str, get_next_version_path, n_iter_per_epoch
from reprlearn.data.datamodules import MNISTDataModule
# Models
from reprlearn.models.conv_generator import ConvGenerator
from reprlearn.models.fc_discriminator import FCDiscriminator
from reprlearn.models.plmodules.gan import GAN
# Callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from reprlearn.callbacks.recon_logger import ReconLogger
from reprlearn.callbacks.beta_scheduler import BetaScheduler
# Evaluations
from reprlearn.models.plmodules.utils import get_best_ckpt_path, get_best_k_ckpt_paths
from reprlearn.models.plmodules.utils import save_sample_from_model_ckpt


# Select Visible GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set Path
# - DATA_ROOT:
#   - Use '/data/hayley-old/Tenanbaum2000/data' for MNIST, Mono-MNIST, Rotated-MNIST, Teapots
#   - Use `/data/hayley-old/maptiles_v2' folder for Maptile dataset
DATA_ROOT = Path("/data/hayley-old/Tenanbaum2000/data")
# DATA_ROOT = Path("/data/hayley-old/maptiles_v2/")
print("DATA_ROOT: ", str(DATA_ROOT))

# For reproducibility, set seeds:
# seed = 100
# pl.seed_everything(seed)
# # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
# model = Model()
# trainer = pl.Trainer(deterministic=True)


##############################################################################
# Experiment starts here
##############################################################################
# Init Datamodule
# use PL.DataModule object
in_shape = (1,32,32)
bs = 1028
num_workers = 8
dm = MNISTDataModule(data_root=DATA_ROOT,
                     in_shape=in_shape,
                     batch_size=bs,
                     pin_memory=True,
                     num_workers=num_workers,
                     shuffle=True)
dm.setup()

# Init LightningModule
# Generator
latent_dim = 10
latent_emb_dim = 32
decoder_dims = [256, 128, 64, 32]
dec_type = 'conv'
# dec_type = 'resnet'
gen = ConvGenerator(latent_dim=latent_dim,
                    latent_emb_dim=latent_emb_dim,
                    dec_hidden_dims=decoder_dims,
                    in_shape=in_shape,
                    dec_type='conv'
                    )

# Discriminator
discr = FCDiscriminator(in_shape=in_shape)
print('Dicr: ', discr.name)

#GAN
lr_g = 1e-4
lr_d = 1e-3
niter_D_per_G = 1
b1 = 0.5
b2 = 0.99
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
max_epochs = 500
max_iters = n_iter_per_epoch(dm.train_dataloader()) * max_epochs

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
# callbacks = [ckpt_callback, early_stopping_callback]


# Init. Tensorboard logger
exp_name = f'{model.name}_{dm.name}'
tb_logger = pl_loggers.TensorBoardLogger(
    save_dir=f'{ROOT}/lightning_logs/{today2str()}',
    name=exp_name,
    default_hp_metric=False, #todo: what is this param's effect?
)

log_dir = Path(tb_logger.log_dir)
print(tb_logger.log_dir)

if not log_dir.exists():
    log_dir.mkdir(parents=True)
    print("Created: ", log_dir)


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
#
# trainer = pl.Trainer(fast_dev_run=3) # for test/debug
trainer = pl.Trainer(**trainer_config) # for real training

# Set logger to log mesaages to a file "train.log"
log_fp = log_dir/"train.log"
logging.basicConfig(filename=log_fp,
                    filemode='w',
                    format='%(asctime)s -- %(message)s',
                    datefmt="%m/%d/%Y %I:%M%S %p",
                    level=logging.DEBUG)
logging.info(f'\nNew train started: now2str()')
logging.info(f"Model: {model.name}")
logging.info(f"\bGenerator: {gen.name}")
logging.info(f"\bDicriminator: {discr.name}")


# Fit model
trainer.fit(model, dm)
logging.info(f"Finished training at ep {trainer.current_epoch}")


##############################################################################
# Log experiment info to Tensorboard
# - model init parameters (ie. `model.hparams`)
# - `best_score` (from val loss?) to Tensorboard
################################################################################
hparams = model.hparams.copy()
hparams.update(dm.hparams)
# hparams['use_beta_scheduler'] = use_beta_scheduler
hparams['ckpt_metric'] = ckpt_metric
hparams['stop_metric'] = stop_metric

best_ckpt_score = trainer.checkpoint_callback.best_model_score.item()
metrics = {f'hparam/best_{ckpt_metric}': best_ckpt_score}
# alternatively, we can log multiple metrics that we kept records of during the training
# metrics['hparam/this_acc']: this_acc # classification accuracy (val) with the model that achieved the best_{ckpt_metric} value
# metrics['hparam/this_pr']: this_pr
# or, best metrics achieved with this hyperparams ("hp") -- though different model states
# metrics['hparam/best_acc'] = 0.999
# metrics['hparam/dummy']= 1.
#hmm actually, ^this may not work as expected because the value added like this,
#shows up only at "SCALARS" section, and not under HPARAMS

logging.info('Start logging hparams and metrics to TenborBoard...')
logging.info('Model, Data params: ')
pprint(hparams)

logging.info('Checkpt metric: ')
pprint(metrics)

# Log to Tensorboard
trainer.logger.log_hyperparams(hparams, metrics)


################################################################################
# Evaluation
# Steps:
# - Define model architecture aka. model's "skeleton", if not done before
# - Load the saved weights for this model architecture from checkpt filepaths
#   into the model skeleton
# - Generate N datapoints
#   - Visually inspect it vs. MNIST
#   - Quantify using mmd
################################################################################
# Generate sample sets from each of the k checkpoints
# 1. Define model skeleton
# model = BetaVAE (...)
n_samples = 10000
sample_dir = Path(trainer.log_dir)/'samples'
if not sample_dir.exists():
    sample_dir.mkdir()
    print('Created: ', sample_dir)

# 2. Get best k ckpts from trainer
k_ckpts = get_best_k_ckpt_paths(trainer)
for ckpt_path in k_ckpts:
    save_sample_from_model_ckpt(
        model,
        ckpt_path,
        DEVICE,
        sample_dir,
        n_samples,
        out_fn_prefix='MNIST',
   )


print('Done saving samples.')


# ### Evaluate the generated sample

def get_sample_fp(sample_dir: Path,
                  ckpt_path: Union[Path,str],
                  prefix: Optional[str]=None) -> Path:
    """Returns the file-path to the pickled file of sample,
    generated from the model at ckpt_path state"""
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    prefix = '' if prefix is None else f'{prefix}-'
    fp = sample_dir/f'{prefix}{ckpt_path.stem}.pkl'
    return fp


# show from all ckpt models and save as snapshot of each of the generated sample sets
from reprlearn.utils.misc import change_suffix

n_show = 100
inds = np.random.randint(0, len(sample), size=n_show)
for ckpt_path in k_ckpts:
    sample_fp = get_sample_fp(sample_dir, ckpt_path, prefix="MNIST")
    assert sample_fp.exists(), f"Sample file doesn't exist: {sample_fp}"
    png_fp = change_suffix(sample_fp, '.png')

    # Load the sample
    sample = torch.load(sample_fp) # loaded to cuda
    sample = sample.cpu()
    print('Loaded sample set: ', sample.shape)

    f, _ = show_timgs(sample[inds], cmap='gray', title=sample_fp.stem); # todo: save as image
    f.tight_layout()
    f.savefig(png_fp)
    print('Saved snapshot of samples: ', png_fp)



