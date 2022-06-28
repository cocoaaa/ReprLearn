import argparse
from argparse import ArgumentParser
import torch
import torch.nn as nn
from .types_ import *
from collections import OrderedDict
from typing import Callable, List, Union, Tuple, Optional
from .utils import make_fc_block, compute_ks_for_conv2d

class FCGenerator(nn.Module):
    def __init__(self, *,
                 latent_dim: int,
                 latent_emb_dim: int,
                 dec_hidden_dims: List[int],
                 in_shape: Union[torch.Size, Tuple[int, int, int]],
                 act_fn: nn.Module = nn.LeakyReLU(),
                 use_bn: bool= False,
                 out_fn: nn.Module = nn.Tanh()
                 ):
        """Generator network with either conv2dTranspose blocks or as ResNet blocks
        Args
        ----
        latent_dim : dim of noise input (z)
        latent_emb_dim : dim of the embedding of the latent noise (z --> fully-connect --> emb_z)
        dec_hidden_dims : number of filters in the main conv layers of the decoder/generator
        in_shape : (nc, h, w) of input datapt
        # dec_type : decoder layer type. 'conv' or 'resnet'. If 'resnet' we use skip connections
        act_fn : each layer's activation function. Default nn.LeakyReLU
        out_fn : final operation's output function. Default nn.Tanh

        Forward-pass flow
        ------------------
        z (bs, latent_dim)
        #1. extend a channel dimension) and "embedding" layer for the latent code vectors
        --> nn.Linear(latent_dim, latent_emb_dim) + act_fn (+ batch_norm if self.use_bn)
        #2. pass through main decov layers
        --> nn.Linear(in_features=latent_emb_dim, out_features=decoder_dims[0]) + act_fn (+ batch_norm if self.use_bn)
        --> nn.Linear(decoder_dims[0], decoder_dims[1]) + act_fn (+ batch_norm if self.use_bn)
        --> ...
        --> nn.Linear(decoder_dims[-1], in_shape[-1]) (+ batch_norm if self.use_bn) # output dimension is in_shape[-1]

        """
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_emb_dim = latent_emb_dim # dim of embedding_z (h_z)
        self.in_shape = in_shape  # in order of (nc, h, w) if x is a 3dim r.v., or (h,w) if 2dim
        self.data_dim = len(in_shape)

        if self.data_dim == 2:
            self.in_h, self.in_w = in_shape
            self.out_dim = self.in_shape[-1]

        elif self.data_dim >= 3 :
            self.in_channels, self.in_h, self.in_w  = in_shape
            self.out_dim = self.in_channels
        else:
            raise ValueError("Currently, this generator supports data_dim of 2 or 3")
        self.dec_type = 'fc'
        self.act_fn = act_fn
        self.use_bn = use_bn
        self.out_fn = out_fn

        # head layer
        self.fc_latent2flatten = nn.Linear(self.latent_dim, self.latent_emb_dim)

        # main generator body (layers of fully-connected weights)
        self.n_feats = [latent_emb_dim, *dec_hidden_dims]
        layers = [make_fc_block(in_, out_, self.act_fn, self.use_bn) \
                  for (in_, out_) in zip(self.n_feats, self.n_feats[1:])]
        self.decoder = nn.Sequential(
            *layers,
            nn.Linear(self.n_feats[-1], self.out_dim)
        )

        # out layer
        if len(self.in_shape) == 2:
            self.out_layer = self.out_fn
        elif len(self.in_shape) == 3:
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
                self.out_fn) #todo: why is this using conv2d layer and not fully-connect/linear layer?

        else:
            raise ValueError("Generator only supports output of 2 or 3-dim")

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
        # print('after latent2flatten: ', result.shape)  # (BS, dim_embedding_z), ie. (BS, self.latent_emb_dim)

        if self.data_dim == 3:
            # expand the latent_code vector to 3dim tensor
            result = result.view(bs, self.latent_emb_dim, 1, 1)
            # print('after 3d expansion: ', result.shape);

        # pass through main decoder layers
        result = self.decoder(result);
        # print('after decoder: ', result.shape);

        # apply output function
        x_gen = self.out_layer(result);
        # print('after out-layer: ', result.shape);
        return x_gen

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []

        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        # parser.add_argument('--in_shape', nargs=3,  type=int, required=True)
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--latent_emb_dim', type=int, required=True,
                            help='dim of embedding of noise vector') # Default None, which is set to 3*latent_dim
        parser.add_argument('--dec_hidden_dims', nargs="+", type=int)  # None as default
        parser.add_argument('--dec_type', type=str, default="conv",
                            help='type of layers, conv or resnet (if wanted with skip)')
        parser.add_argument('--gen_act_fn', type=str, default="leaky_relu",
                            help="Choose relu or leaky_relu (default)")  # todo: proper set up in main function needed via utils.py's get_act_fn function
        parser.add_argument('--gen_out_fn', type=str, default="tanh",
                            help="Output function applied at the output layer of the decoding process. Default: tanh")

        return parser




class VecDecoder(nn.Module):
    """Fully-connected layers with input vector of size `z_dim` 
    and output vector of size `target_dim`
    """
    def __init__(self, 
                 input_dim: int, #z_dim
                 nfs: List[int], # number of units in hidden units
                 out_dim: int, #target_dim
                 act_fn: Optional[Callable]=nn.LeakyReLU(),
                 out_fn: Optional[Callable]=nn.Identity(),
                ):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        nfeats = [self.input_dim, *nfs]
        layers =  [make_fc_block(in_, out_, act_fn=act_fn) \
                  for (in_, out_) in zip(nfeats, nfeats[1:])]
        last_layer = make_fc_block(nfeats[-1], self.out_dim, act_fn=out_fn)
        self.decoder = nn.Sequential(
            *layers,
            last_layer
        )
        
    def forward(self, z):
        """  z --> y
        input: z (code vector) 
        output: y (target vector)
        """
        return self.decoder(z)
    
    