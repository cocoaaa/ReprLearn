from .types_ import *
from copy import deepcopy
from argparse import ArgumentParser
import torch
from .gan import GAN
from ..conv_generator import ConvGenerator
from ..fc_discriminator import FCDiscriminator

class ConvFCGAN(GAN):
    """Sets generator to be ConvGenerator, discriminator to be FCDiscriminator.
    Pass all of its input parameters to GAN class as is."""
    def __init__(self, *,
                 in_shape: Union[torch.Size, Tuple[int, int, int]],
                 latent_dim: int,
                 latent_emb_dim: Optional[int] = None, # Default to 3*latent_dim
                 dec_type: str = 'conv', #'conv' or 'resnet',
                 dec_hidden_dims: Optional[List[int]] = None, #nfs of main conv-layers
                 # discr_hidden_dims: List[int], #todo: add this feature
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
        gen = ConvGenerator(
            latent_dim=latent_dim,
            latent_emb_dim=latent_emb_dim,
            dec_type=dec_type,
            dec_hidden_dims=dec_hidden_dims,
            in_shape=in_shape,
        )
        discr = FCDiscriminator(in_shape=in_shape)

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


    @property
    def name(self) -> str:
        bn = "ConvFCGAN"
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
        gen_p = ConvGenerator.add_model_specific_args(parent_parser)
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


