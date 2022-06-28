from .types_ import *
from ml_collections import ConfigDict
from copy import deepcopy
from argparse import ArgumentParser
import torch
import torch.nn as nn
from .gan import GAN
from ..fc_generator import FCGenerator
from ..fc_discriminator import FCDiscriminator


class FCFCGAN(GAN):
    """Sets generator to FCGenerator, discriminator to FCDiscriminator.
    Pass all of its input parameters to GAN class as is."""
    def __init__(self, 
                 in_shape: Union[torch.Size, Tuple[int, int, int]],
                 latent_dim: int,
                 latent_emb_dim: Optional[int] = None, # Default to 3*latent_dim
                 dec_hidden_dims: Optional[List[int]] = None, #n_features for generator's fc layers
                 dec_out_fn: Optional[Callable] = None, #output layer of generator
                 discr_hidden_dims: List[int] = None, #todo: add this feature
                 lr_g: float = 1e-4,
                 lr_d: float = 1e-3,
                 niter_D_per_G: int = 1,
                 # label_map: Optional[Dict] = None,
                 size_average: bool = False,
                 log_every: int = 10, #log generated images every this epoch
                 n_show: int = 100,  # number of images to log at checkpt iteration
                 **kwargs,
                 ):
        latent_emb_dim = latent_emb_dim or 3*latent_dim
        dec_hidden_dims = dec_hidden_dims or [256, 128, 64, 32]
        dec_out_fn = dec_out_fn or nn.Identity() # e.g., nn.Sigmoid(), nn.Tanh()
        discr_hidden_dims = discr_hidden_dims or [256, 128, 64]
        gen = FCGenerator(
            latent_dim=latent_dim,
            latent_emb_dim=latent_emb_dim,
            dec_hidden_dims=dec_hidden_dims,
            in_shape=in_shape,
            out_fn=dec_out_fn,
        )
        discr = FCDiscriminator(
            in_shape=in_shape,
            discr_hidden_dims=discr_hidden_dims,
        )
    
        super().__init__(
            in_shape=in_shape,
            latent_dim=latent_dim,
            generator=gen,
            discriminator=discr,
            lr_g=lr_g,
            lr_d=lr_d,
            niter_D_per_G=niter_D_per_G,
            size_average=size_average,
            log_every=log_every,
            n_show=n_show,
            **kwargs,
        )

    # def __init__(self, config: ConfigDict):
    #     gen_config = {
    #         'in_shape': config.data.in_shape, #data variable x.shape
    #         'latent_dim': config.model.gen.latent_dim, #input noise variable z's dim
    #         'latent_emb_dim': config.model.gen.latent_emb_dim,
    #         'dec_hidden_dims': config.model.gen.dec_hidden_dims, #1 embedding layer + 2 intermediate layers + 1 output layer
    #         'act_fn': config.model.gen.act_fn,
    #         'use_bn': config.model.gen.use_bn,
    #         'out_fn': config.model.gen.out_fn, # any real values (for x1, x2 coordinates); no squashing to certain range,
    #     }

    #     discr_config = {
    #         'in_shape': config.data.in_shape, # data-variable is a vector in 2dim
    #         'discr_hidden_dims': config.model.discr.discr_hidden_dims,
    #     }

    #     gen = FCGenerator(**gen_config)
    #     discr = FCDiscriminator(**discr_config)

    #     super().__init__(
    #         in_shape=config.data.in_shape,
    #         latent_dim=config.model.gen.latent_dim,
    #         generator=gen,
    #         discriminator=discr,
    #         lr_g=config.optim.lr_g,
    #         lr_d=config.optim.lr_d,
    #         niter_D_per_G=config.train.niter_D_per_G,
    #         # size_average=config.train.size_average,
    #         log_every=config.logging.log_every,
    #         n_show=config.logging.num_samples,
    #         b1=config.optim.beta1,
    #         b2=config.optim.beta2
    #     )


    @property
    def name(self) -> str:
        bn = "FCFCGAN"
        return (
            f'{bn}'
            f'-{self.generator.name}-{self.discr.name}'
            f'-dimz:{self.latent_dim}'
            f'-lr_g:{self.lr_g}-lr_d:{self.lr_d}'
            f'-K:{self.niter_D_per_G}'
            f'-b1:{self.b1}-b2:{self.b2}'
        )


    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []
        base_p = GAN.add_model_specific_args(parent_parser)
        gen_p = FCGenerator.add_model_specific_args(parent_parser)
        discr_p = FCDiscriminator.add_model_specific_args(parent_parser)
        parents = [base_p, gen_p, discr_p]

        # Get the parser with base class's arguments added
        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        parser.add_argument('--lr_g', type=float, required=True,
                            help="Learn-rate for generator")
        parser.add_argument('--lr_d', type=float, required=True,
                            help="Learn-rate for discriminator")
        parser.add_argument('-k', '--niter_D_per_G', type=int, required=True,
                            dest='niter_D_per_G',
                            help="Update D every iteration and G every k iteration")
        return parser



