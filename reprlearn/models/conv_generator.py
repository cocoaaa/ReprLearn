import torch
import torch.nn as nn
from .types_ import *

from collections import OrderedDict
from typing import Callable, List, Union, Tuple, Optional
from reprlearn.models.convnet import conv_blocks, deconv_blocks
from reprlearn.models.resnet import ResNet
from reprlearn.models.resnet_deconv import ResNetDecoder
from .utils import compute_ks_for_conv2d

class ConvGenerator(nn.Module):
    def __init__(self, *,
                 latent_dim: int,
                 len_flatten: int,
                 decoder_dims: List[int],
                 in_shape: Union[torch.Size, Tuple[int, int, int]],
                 dec_type: str = 'conv',  # 'conv' or 'resnet'
                 act_fn: Callable = nn.LeakyReLU(),
                 out_fn: Callable = nn.Tanh()
                 ):
        """Generator network with either conv2dTranspose blocks or as ResNet blocks

        z (bs, latent_dim)
        #1. extend a channel dimension) and "embedding" layer for the latent code vectors
        --> nn.Linear(latent_dim, len_flatten)
        #2. pass through main decov layers
        --> conv2dTranspose(in_channels=1,out_channels=decoder_dims[0],kernel...)
        --> conv2dTranspose(in_channels=decoder_dims[1], out_channels=decoder_dims[2], kernel..)
        --> ...
        --> conv2dTranspose(in_channels=decoder_dims[-1], out_channels=in_shape[0], kernel...) # out_channels is image's nC

        """
        super().__init__()
        self.latent_dim = latent_dim
        self.len_flatten = len_flatten # dim of embedding_z (h_z)
        self.in_shape = in_shape  # in order of (nc, h, w)
        self.in_channels, self.in_h, self.in_w = in_shape
        self.dec_type = dec_type
        self.act_fn = act_fn
        self.out_fn = out_fn

        # head layer
        self.fc_latent2flatten = nn.Linear(self.latent_dim, self.len_flatten)

        # main decov layer
        self.nfs = [len_flatten, *decoder_dims, self.in_channels]
        if self.dec_type == 'conv':
            self.decoder = deconv_blocks(self.nfs[0],
                                       self.nfs[1:],
                                       has_bn=True,
                                       act_fn=act_fn)
        elif self.dec_type == 'resnet':
            self.decoder = ResNetDecoder(self.nfs, act_fn=act_fn)  # todo?

        # out layer
        # outputs the same (h,w) as the input's (h,w)
        self.n_deconv_blocks = len(self.nfs) - 1
        # compute last feature map's spatial dims
        self.last_h = 2 ** self.n_deconv_blocks
        self.last_w = 2 ** self.n_deconv_blocks
        print('last h,w: ', self.last_h, self.last_w)

        last_k_h = compute_ks_for_conv2d(w_in=self.last_h, w_out=self.in_h)
        last_k_w = compute_ks_for_conv2d(w_in=self.last_w, w_out=self.in_w)
        print('last_kernel h,w: ', last_k_h, last_k_w)
        self.out_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=(last_k_h, last_k_w), stride=1, padding=1),
            self.out_fn)

    @property
    def name(self):
        bn = 'Decoder'
        return f'{bn}-{self.dec_type}'

    def forward(self, z: Tensor) -> Tensor:
        """ Maps the given latent code onto the image space.
       :param z: (Tensor) [B x latent_dim]
       :return: (Tensor) [B x C x H x W]: generated images
       """

        bs = len(z)
        # learn embedding of the latent code vector, h_z
        result = self.fc_latent2flatten(z);
        print('after latent2flatten: ', result.shape)  # (BS, dim_embedding_z), ie. (BS, self.len_flatten)

        # expand the latent_code vector to 3dim tensor
        result = result.view(bs, self.len_flatten, 1, 1);
        print('after 3d expansion: ', result.shape);

        # pass through main decoder layers
        result = self.decoder(result);
        print('after decoder: ', result.shape);

        # apply output function
        x_gen = self.out_layer(result);
        print('after out-layer: ', result.shape);
        return x_gen

