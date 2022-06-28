#!/usr/bin/env python
# coding: utf-8

"""
Latent sampling of GAN, guided by the geometry of GAN's data manifold
- latent_dim: (z_dim) is 1
- data_dim: 2dim (1dim data manifold (aka. curve/line) embedded in 2d space

Train 1d gan as suggested by Greg
1. collect training dataset -- use hololens? library
2. train gan
3. sample from z's and color-code the mapped x's = G(z) with the color as value of z
4. make the plot of the arclength

Changelog
- 2022-04-19: turn this into a run_lib.py that contains the main train function
later i'll add evaluation function as well.
this run_lib.py is used in the main.py

"""


import os,sys
from datetime import datetime
import time
from collections import OrderedDict
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any,List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
from pprint import pprint

from absl import flags

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
# Callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from reprlearn.callbacks import LogScatterPlot2DiskCallback, LogScatterPlot2TBCallback

import reprlearn as rl
from reprlearn.visualize.utils import show_timg, show_timgs, show_batch, make_grid_from_tensors, set_plot_lims
from reprlearn.utils.misc import info, now2str, today2str, get_next_version_path, n_iter_per_epoch
from reprlearn.models.utils import inplace_freeze

# Select Visible GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global vars
ROOT = Path('/data/hayley-old/Proj2')
FLAGS = flags.FLAGS


# true data manifold (1dim, embedded in 2dim x-space)
n_sample = 10_000
x1s = np.linspace(-5, 5, num=n_sample)
x2s = 0.5*x1s**3 - 10*x1s
X_cts = np.c_[x1s, x2s]
plt.plot(X_cts[:,0], X_cts[:,1])
plt.title("Continuous true manifold")
plt.xlabel('x1')
plt.ylabel('x2')

# plt.savefig(fname="./cts-true-data-manifold.png")


# Disconnected manifold to be used as training dataset
# 3 submanifold (sets of x's), each from
is_in_submani1 = (x1s <= -2.5)
is_in_submani2 = (-1 <= x1s) & (x1s <=1)
is_in_submani3 = (3 <= x1s)


# In[ ]:


submani1 = X_cts[is_in_submani1]
submani2 = X_cts[is_in_submani2]
submani3 = X_cts[is_in_submani3]

print([len(s) for s in [submani1, submani2, submani3]])

# let's make them all contain same number of samples, just in case
n_sample_per_submani = min(map(len, [submani1, submani2, submani3]))
print('n_sample_per_submani :', n_sample_per_submani)

submani1, submani2, submani3 = map(
    lambda s: s[:n_sample_per_submani],
    [submani1, submani2, submani3]
)

print('---')
print('Disconnected manifold to train our GAN')
print([len(s) for s in [submani1, submani2, submani3]])
X_data = np.r_[submani1, submani2, submani3]
print('X_train shape: ', X_data.shape)
print('Each submani shape: ', submani1.shape)


# plot
f, ax = plt.subplots()
ax.plot(X_cts[:,0], X_cts[:,1], 'k--')
ax.scatter(X_data[:,0], X_data[:,1])
ax.set_title('Disconnected manifold')
f.savefig(f'./true-disconnected-manifold-{no2str()}.png')




# Fit GAN to this training dataset
from torch.utils.data import TensorDataset

# shuffle X_train's elements; split X_train to train, val, test
np.random.shuffle(X_data)
print(X_data.shape, X_data[0])

n_data = len(X_data)
# split to train, val, test = 7:2:1
n_train, n_val = int(n_data*0.7), int(n_data*0.2)
n_test = n_data - (n_train + n_val)

X_train = X_data[:n_train]
X_val = X_data[n_train:n_train+n_val]
X_test = X_data[-n_test:]

dset_train =  TensorDataset(torch.Tensor(X_train))
dset_val = TensorDataset(torch.Tensor(X_val))
dset_test = TensorDataset(torch.Tensor(X_test))

dsets = {
    'train': dset_train,
    'val': dset_val,
    'test': dset_test
}


x = dset_train[10][0]
x, X_train[10]


# helper: plt np.array of data from each Dataset instances
def plot_datasets(dsets: Dict[str,Union[torch.Tensor,np.ndarray]],
                  colors: Optional[Dict[str, str]]=None,
                  figsize: Optional[Tuple]=None,
                  **kwargs,
                  ) -> Tuple[plt.Figure, plt.Axes]:
    """Plot train, val, test datasets' underlying np.array data"""
    if colors is None:
        colors = {
            'train': 'r',
            'val': 'g',
            'test': 'b'
        }

    figsize = figsize or (12,3)

    f, axes = plt.subplots(1,3, figsize=figsize)
    axes = axes.flatten()
    i = 0
    for (mode, data) in dsets.items():
        axes[i].scatter(data[:,0], data[:,1], c=colors[mode], **kwargs)
        axes[i].set_title(mode)
        i += 1
    return f, axes


# In[ ]:


# test: plot each train, val, test datasets' data
colors = {
    'train': 'r',
    'val': 'g',
    'test': 'b'
}
f, axes = plt.subplots(1,3, figsize=(12,3))
axes = axes.flatten()
i = 0
for (mode, dset) in dsets.items():
    data = dset.tensors[0].numpy()
    axes[i].scatter(data[:,0], data[:,1], c=colors[mode])
    axes[i].set_title(mode)
    i += 1


# ### Init. DataModule

# In[ ]:


from pytorch_lightning import LightningDataModule, Trainer
from reprlearn.data.datamodules.base_datamodule import BaseDataModule


# In[ ]:


# define a simple LightningDataModule class for TensorDataset

class TensorDataModule(BaseDataModule):
    def __init__(self,
                 X_train: Tensor,
                 X_val: Tensor,
                 X_test: Optional[Tensor] = None,
                 dm_name: Optional[str]='tensor-dm',
                 **kwargs):
        """LightingDataModule that wrapps TensorDataset where the underlying data
        is a simple 2dim or 3dim tensor (rather than, e.g., images).

        Args:
        -----
        X_train : Tensor for training data
        X_val : Tensor for validation data
        X_test : Optional[Tensor]; Tensor for test data
        dm_name: Optional[str]='tensor-dm'

        kwargs:
        -------
        in_shape : Tuple of length 2 (nrows, ncols) or 3 (nc, h, w)
        batch_size : int
        pin_memory : bool. Default True
        num_workers: int. Default 16
        shuffle : bool. True means, shuffle the outputted data at each iter.

        """
        kwargs.update({'data_root': None})
        super().__init__(**kwargs)
        self.dset_train = TensorDataset(X_train)
        self.dset_val = TensorDataset(X_val)
        self.dset_test = TensorDataset(X_test)
        self.dm_name = dm_name

    @property
    def name(self) -> str:
        return self.dm_name

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # assign train/val datasets for use in datalaoders
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.dset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.dset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )



# In[ ]:


in_shape = (1,2)
dm_config = {
    'in_shape': in_shape,
    'batch_size': 64,
}

# [x] test the TensorDataModule:
# - [x] use the data we have now
# - [x] check the values set to teh dm.dset_train/val/test
# - [x] check the values returned from the dm.train/val/test dataloaders

dm = TensorDataModule(
    X_train=torch.tensor(X_train, dtype=torch.float32),
    X_val=torch.tensor(X_val, dtype=torch.float32),
    X_test=torch.tensor(X_test,dtype=torch.float32),
    **dm_config
)


# In[ ]:


# test: TensorDataModule
# -- test1: check if underlying data in each train/val/test datasets
d_train = dm.dset_train.tensors[0] #(x,y) usually, but here we only give 'x's
d_val = dm.dset_val.tensors[0]
d_test = dm.dset_test.tensors[0]
ds = {'train': d_train, 'val': d_val, 'test': d_test}

f, axes = plot_datasets(ds, figsize=(20,5), linewidths=0.1)


# In[ ]:





# In[ ]:


# test2: TensorDataModule
# -- test if the train/val/test dataloaders are working properly
colors = {
    'train': 'r',
    'val': 'g',
    'test': 'b'
}

dls = {
    'train': dm.train_dataloader(),
    'val': dm.train_dataloader(),
    'test': dm.test_dataloader()
}


for i, (mode, dl) in enumerate(dls.items()):
    batch_x = next(iter(dl))[0]
    print(mode, len(batch_x))
    axes[i].scatter(batch_x[:,0], batch_x[:,1],
                    c=colors[mode],
                    #                     edgecolors='y',
                    marker='*',
                    linewidths=10.)

f


# ### Init. Model
# GAN with fc-generator, fc-discriminator

# In[ ]:


# set the loss function of the generator to be euclidean dist
from reprlearn.models.fc_generator import FCGenerator
from reprlearn.models.fc_discriminator import FCDiscriminator
from reprlearn.models.plmodules.fc_fc_gan import FCFCGAN


# In[ ]:


# test fc_generaetor.py
gen_config = {
    'in_shape': (1,2), #data variable x.shape
    'latent_dim': 1, #input noise variable z's dim
    'latent_emb_dim': 2,
    'dec_hidden_dims': [100, 100, 100], #1 embedding layer + 2 intermediate layers + 1 output layer
    'act_fn': nn.LeakyReLU(),
    'use_bn': False,
    'out_fn': nn.Identity() # any real values (for x1, x2 coordinates); no squashing to certain range,
    'lr_g': 5e-4,
}

# gen = FCGenerator(**gen_config)


# In[ ]:


# Discriminator
discr_config = {
    'in_shape': (1,2), # data-variable is a vector in 2dim
    'discr_hidden_dims': [100, 100, 100],
    'lr_d': 5e-4
}
# discr = FCDiscriminator(**discr_config)


# In[ ]:


# Init. GAN
in_shape = (1,2) #fixed! data-variable's dimension; (x1,x2) is our datapt
config = {**gen_config, **discr_config}
model = FCFCGAN(**config)


# ### Init. Trainer

# #### Set callbacks
#

# In[ ]:




# In[ ]:


max_epochs = 1000
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


# In[ ]:


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


# Init. disk writer to save images of generated sample to disk
log_every = 10
num_samples = 300
sample_dir = log_dir/"Samples"
if not sample_dir.exists():
    sample_dir.mkdir(parents=True)
    print("Created: ", sample_dir)

plot_xlim = (-5, 5) #todo: don't hardcode?
disk_plot_logger = LogScatterPlot2DiskCallback(
    log_every = log_every,
    num_samples = num_samples,
    save_dir = sample_dir,
    xlim=plot_xlim
)

tb_plot_logger = LogScatterPlot2TBCallback(
    log_every=log_every,
    num_samples=num_samples,
    xlim=plot_xlim
)


# Final callbacks
# callbacks = [ckpt_callback]
# callbacks = [ckpt_callback, early_stopping_callback]
callbacks = [ckpt_callback, disk_plot_logger, tb_plot_logger]


# In[ ]:


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


# ### Start training
#

# In[ ]:


# Start experiment
# Fit model
trainer.fit(model, dm)
print(f"Finished at ep {trainer.current_epoch}")


# In[ ]:





# In[ ]:





# In[ ]:





# ## Evaluation
# - sample multiple points from z and get x's mapped by G(z)
# - scatter plot these x's with color

# In[ ]:


model


# In[ ]:





# ### Load ckpt
#

# In[ ]:


# replace '...' witht ckpt filename to load model states
# target_ckpt_fp = ckpt_dir / "..."
# model.load(target_ckpt_fp)


# ### Color the scatterplot of x's with input z values

# In[ ]:


# first sample many z's
model.to(DEVICE) #inplace
model.eval()
n_eval_samples = 1_000
z_samples, x_samples = model.sample(n_eval_samples, current_device=DEVICE, return_z=True)
z_samples, x_samples = z_samples.cpu(), x_samples.cpu()


# In[ ]:


# create custom cmap to show linearly in z
import matplotlib as mpl


# In[ ]:


cmap = plt.cm.get_cmap('jet', len(z_samples))
plt.scatter(x=z_samples[:,0],
            y=[1]*len(z_samples),
            c=z_samples, #imp!
            cmap=cmap,   #imp!
            linewidths=0.1
            )
plt.colorbar()
plt.title('z ~ N(0,1) with color-id')


# In[ ]:


# now, we use the color of z's to color-code its matching x's
plot_xlim = (-5,5)
plot_ylim = (-5,5)

f_eval, ax_eval = plt.subplots( )#figsize=(12,3))
ax_eval.scatter(x=x_samples[:,0],
                y=x_samples[:,1],
                c=z_samples, #imp!
                cmap=cmap,   #imp!
                linewidths=0.1
                )
set_plot_lims(ax_eval, xlim=plot_xlim, ylim=plot_ylim)
ax_eval.set_title('x in 2dim')
ax_eval.set_xlabel('x1')
ax_eval.set_ylabel('x2')







# ### Compute arclength in data space
# - make sure the model weights are frozen (requires_grad = False)
# - what we want to do, is to backprop the gradient from x = G(z) to the input variable z
# - that means, we need to have the requires_grad of input z's to be True
#

# woohoo, finally working
# supports multivariate input (batch), multivariate output
# input x: torch.Size([2, 3])
# output y: torch.Size([2, 5])
# jacobian dy/dx: torch.Size([2, 5, 3])  b, out_dim, in_dim
# src: https://tinyurl.com/y3aono4m

def get_batch_jacobian(net, x, noutputs):
    """
    input: (bs, input_dim)
    output: (bs, out_dim)

    Returns
    -------
    jacobian : jacobian[b][out_dim_i][input_dim_j]
    """
    #     breakpoint()
    x = x.unsqueeze(1) # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1) # b, out_dim, in_dim
    x.requires_grad_(True)
    x.retain_grad()

    y = net(x)
    input_val = torch.eye(noutputs, device=y.device).reshape(1,noutputs, noutputs).repeat(n, 1, 1)

    y.backward(input_val)
    return x.grad.data


# - instantaneous arclength of x = G(z) is the norm of this gradient vector
def compute_arclength(z: torch.Tensor,
                      push_forward_fct: Callable,
                      device: torch.device) -> torch.Tensor:
    """ Given an input variable z and push_forward function (e.g. Genenerator),
    compute the norm of the gradient G(z) wrt z.
    This norm can be interpreted as the instantaneous arclength of x in data space

    Background:
    -----------
    z --> G --> x
    Here, all weights of G is frozen.
    Our goal is to compute grad_z(x), the gradient vector of x wrt z.

    - Key is to make sure z is set to `require_grad` and to `retain_grad`
    We don't enforce it to the caller side, and rather make sure this condition
    is met by running the commands inside this function.
    - Similarly, we make sure all the weights are turned off from auto-diff's
    gradient tracking


    Returns:
    -------
    arclength : Tensor; tensor whose ith element is the arclength of x[i] mapped
                from z[i]

    """
    batch_grad = get_batch_jacobian(push_forward_fct, z, data_dim)
    # z.grad now has the (vector) gradient
    arclens = [torch.norm(grad, dim=0).cpu() for grad in batch_grad]
    return arclens

