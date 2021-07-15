from pprint import pprint
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load


def compute_kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute the expected/averaged kl-divergence from a batch of mu's and logvar's
    for a fully-factorized Gaussian distribution.

    Assume dim0 is the batch dimension, and all the rest are data dimension.

    .. note:: this function handles mu,logvar tensors whose dim > 2 (eg. mu_z maintains latent factors
    for each channel, and thus have a shape of (nChannel=3, embedding_height=10, embedding_width=10).

    :param mu: (BS, ...)
    :param logvar: (BS, ...)
    :return: a float tensor of average  kl-divergence per datapoint
    """
    batch_dim = 0
    data_dims = tuple(range(1, mu.ndim))
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=data_dims),
                     dim=batch_dim)
    return kld


def get_best_ckpt(model: pl.LightningModule,
                  verbose:bool = False):
    ckpt_path = model.trainer.checkpoint_callback.best_model_path

    if verbose:
        ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)  # dict object
        for k, v in ckpt.items():
            if 'state' in k:
                continue
            if isinstance(v, dict):
                pprint(f"{k}")
                pprint(f"{v}")
            else:
                pprint(f"{k}:{v}")
    return ckpt_path

def load_model(model: pl.LightningModule, ckpt_path: str):
    # Inplace loading of the model state from the ckpt_path
    # ckpt_path = get_best_ckpt((model))
    ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)  # dict object
    model.load_state_dict(ckpt['state_dict'])

def load_best_model(model: pl.LightningModule):
    """
    Load the model state from the best ckpt_path recorded during the training
    Update the model's state **inplace**

    :param model: pl.LightningModule
    :return:
    """
    ckpt_path = get_best_ckpt((model))
    ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)  # dict object
    model.load_state_dict(ckpt['state_dict'])