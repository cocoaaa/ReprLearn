from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar

import torch
import torch.nn as nn

# plmodules
from reprlearn.models.plmodules.three_fcs import ThreeFCs
from reprlearn.models.plmodules.vanilla_vae import VanillaVAE
from reprlearn.models.plmodules.beta_vae import BetaVAE
from reprlearn.models.plmodules.iwae import IWAE
from reprlearn.models.plmodules.bilatent_vae import BiVAE

# datamodules
from reprlearn.data.datamodules import MNISTDataModule
from reprlearn.data.datamodules import MultiMonoMNISTDataModule
from reprlearn.data.datamodules import MultiRotatedMNISTDataModule
from reprlearn.data.datamodules import MaptilesDataModule
from reprlearn.data.datamodules import MultiMaptilesDataModule
from reprlearn.data.datamodules import MultiOSMnxRDataModule

# src helpers
from reprlearn.utils.misc import info


def add_base_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Define general arguments for the command line interface to run the experiment script for training/testing
    Set ArgumentParser to parse
    - model_name
    - data_name
    - gpu_id: ID of GPU to set visible as os.environ
    - mode: fit of test
    - log_root: Root dir to save Lightning logs
    """
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of dataset: mnist, multi_mono_mnist, multi_rotated_mnist, maptiles, multi_maptiles, osmnx_roads")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True, help="ID of GPU to use")
    parser.add_argument("--mode", type=str, default='fit', help="fit or test")
    parser.add_argument("--log_root", type=str, default='/data/hayley-old/Tenanbaum2000/lightning_logs',
                        help='Root directory to save lightning logs')
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser


def get_act_fn(fn_name:str) -> Callable:
    fn_name = fn_name.lower()
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
    }[fn_name]


def get_out_fn(fn_name:str) -> Callable:
    fn_name = fn_name.lower()
    return {
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }[fn_name]


def get_dm_class(dm_name:str) -> object:
    dm_name = dm_name.lower()
    return {
        'mnist': MNISTDataModule,
        'multi_mono_mnist': MultiMonoMNISTDataModule,
        'multi_rotated_mnist': MultiRotatedMNISTDataModule,
        'maptiles': MaptilesDataModule,
        'multi_maptiles': MultiMaptilesDataModule,
        'osmnx_roads': MultiOSMnxRDataModule,
        # TODO: Add new data modules here

    }[dm_name]


def get_model_class(model_name: str) -> object:
    model_name = model_name.lower()
    return {
        "three_fcs": ThreeFCs,
        "vae": VanillaVAE,
        "beta_vae": BetaVAE,
        "iwae": IWAE,
        "bivae": BiVAE,
        # TODO: Add new pl modules here
        # e.g. dcgan, prog_gan, stylegan1, stylegan2

    }[model_name]


def instantiate_dm(args):
    data_name = args.data_name.lower()
    data_root = Path(args.data_root)

    if data_name == 'mnist':
        kwargs = {
            'data_root': data_root,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
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

    elif data_name == 'multi_mono_mnist':
        kwargs = {
            'data_root': data_root,
            'colors': args.colors,
            'seed': args.seed,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
        }
        dm = MultiMonoMNISTDataModule(**kwargs)

    elif data_name == 'multi_rotated_mnist':
        kwargs = {
            'data_root': data_root,
            'angles': args.angles,
            'selected_inds_fp': args.selected_inds_fp,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
            'split_seed': args.split_seed,
        }
        dm = MultiRotatedMNISTDataModule(**kwargs)

    elif data_name == 'multi_maptiles':
        kwargs = {
            'data_root': data_root,
            'cities': args.cities,
            'styles': args.styles,
            'zooms': args.zooms,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
        }
        dm = MultiMaptilesDataModule(**kwargs)

    elif data_name == 'osmnx_roads':
        kwargs = {
            'data_root': data_root,
            'cities': args.cities,
            'bgcolors': args.bgcolors,
            'edge_color': args.edge_color,
            'lw_factor': args.lw_factor,
            'zooms': args.zooms,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
        }
        dm = MultiOSMnxRDataModule(**kwargs)
    # TODO: Add new data modules here

    else:
        raise KeyError(
            "Data name must be in ['mnist', 'maptiles', 'multi_mono_mnist', 'multi_rotated_mnist', 'multi_maptiles', 'osmnx_roads]"
            # TODO: Add new data module's name here
        )

    return dm


def instantiate_model(args):
    # Base init kwargs
    act_fn = get_act_fn(args.act_fn)
    out_fn = get_out_fn(args.out_fn)
    kwargs = {
        'in_shape': args.in_shape, #dm.size()
        'latent_dim': args.latent_dim,
        'hidden_dims': args.hidden_dims,
        'act_fn': act_fn,
        'out_fn': out_fn,
        'learning_rate': args.learning_rate,
        'verbose': args.verbose,
    }

    # Add extra kwargs specific to each model_class
    model_name = args.model_name
    model_class = get_model_class(model_name)
    if model_name == 'beta_vae':
        extra_kw = {
            "kld_weight": args.kld_weight,
            "enc_type": args.enc_type,
            "dec_type": args.dec_type,
        }
        kwargs.update(extra_kw)

    elif model_name == 'iwae':
        kwargs['n_samples'] = args.n_samples

    elif model_name == 'bivae':
        extra_kw = {
            "n_styles": args.n_styles,
            "adversary_dims": args.adversary_dims,
            "is_contrasive": args.is_contrasive,
            "kld_weight": args.kld_weight,
            "adv_loss_weight": args.adv_loss_weight,
            "enc_type": args.enc_type,
            "dec_type": args.dec_type,
        }
        kwargs.update(extra_kw)
    # TODO: Add one for new model here


    return model_class(**kwargs)
