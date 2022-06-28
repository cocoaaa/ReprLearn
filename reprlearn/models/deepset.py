from .types_ import *
import torch
from torch import nn
from reprlearn.models.convnet import conv_blocks
from .utils import make_fc_block

from reprlearn.layers.lambda_layer import Adder, Lambda

class ImageSetEncoder(nn.Module):
    """Abstract class for permutation-invariant function, 
        implemented as a neural network.
        
        f(X) = \rho( \sum_{x^{(n)} \in X}^{B} \phi(x^{(n)}) )
        where X is the input set consisted of B number of data points
        x^{(1)}, ... x^{(B)}.
        
        Thus, this class has two kinds of networks as its properties:
        1. phi: encoder that operates on each data point 
        2. rho: aggregate/pooling network that combines the set of outputs from
        the phi-encoder to make its outcome invariant to the permutations of the output set
        of the phi-encoder.
            Nb. rho-network consists of the aggregator and pooling_encoder
        
        We name the phi and rho networks as:
        (1) phi as instance_encoder, and
        (2) rho as pooling_network 
    """
    def __init__(self, 
                 in_shape: Iterable[int], #input of a set with shape (n_imgs_per_bag, nc, h, w)
                 z_dim: int,
                 nfs_img_encoder: Optional[Iterable[int]] = [300, 100, 30],
                 img_encoder_has_bn: bool= True,
                 img_encoder_act_fn: Callable = nn.LeakyReLU(),
                 pooling_nfeats: Optional[Iterable[int]] = [100, 30],
                 pooling_act_fn: Optional[Callable] = None,
                 pooling_out_fn: Optional[Callable] = nn.Identity(),
                 aggregator_type: Optional[str]='sum', #sum or max
                 aggregator_keepdim: bool=False,
                ):
        
        assert len(in_shape) == 4, "each datapoint must be a set of size (n_imgs_per_bag, nc, h, w)"
        self.n_imgs_per_bag, self.nc, self.h, self.w = in_shape
        
        super().__init__()
        
        # Encoder that operates on each element in the input set
        self.img_encoder = conv_blocks(
            in_channels = self.nc, 
            nf_list = nfs_img_encoder,
            has_bn = img_encoder_has_bn,
            act_fn = img_encoder_act_fn,
        )
        
        # compute last feature map's spatial dims (h_dim)
        self.last_h = self.h // 2 ** len(nfs_img_encoder)
        self.last_w = self.w // 2 ** len(nfs_img_encoder)
        self.h_dim = self.last_h * self.last_w * nfs_img_encoder[-1]
        print('last h,w: ', self.last_h, self.last_w)
        print('flattend h_dim: ', self.h_dim)

        # Pooling network that takes the output (a set) of the encoder
        # to get a permutation-invariant representation
        self.aggregator_type = aggregator_type
        if self.aggregator_type == 'sum':
          self.aggregator = Adder(axis=0, keepdim=aggregator_keepdim) # currently axis=0 because we call aggregator on each set_x  #Implements torch.sum(self.h_dim, axis=1)
        else: # use max-pooling
          # self.aggregator = nn.
          raise ValueError('Currently supports only sum aggregator')
        
        self.z_dim = z_dim
        nfeats_pooling = [self.h_dim, *pooling_nfeats]
        pooling_layers = [make_fc_block(in_, out_, act_fn=pooling_act_fn) \
                  for (in_, out_) in zip(nfeats_pooling, nfeats_pooling[1:])]
        last_pooling_layer = make_fc_block(nfeats_pooling[-1], self.z_dim, act_fn=pooling_out_fn)
        self.pooling_encoder = nn.Sequential(
          *pooling_layers,
          last_pooling_layer
        )
#         self.pooling_network = nn.Sequential(
#             self.aggregator,
#             pooling_encoder
#         )

    
    @property
    def name(self):
        return "SetEncoder"
    
    
    def forward(self, batch_set_x, debug=False) -> torch.Tensor:
      """
      todo: #[[Qs]] is there a way to make the forward_set_x operation natively 
      operate on a batch of set_x? 
      
      Args
      ----
      batch_set_x: a mini-batch whose element is a set
        Shape:  (bs, (n_imgs_per_set, nc, h, w))
      
      Returns
      -------
      batch_z : (bs, z_dim)
        Each z is a z_dim vector, correpsonding to each input of set_x in the batch_set_x
      
      """
      #todo
      batch_z = [ ] 
      for set_x in batch_set_x: 
        batch_z.append(self.forward_set_x(set_x, debug=debug))
        
      return torch.stack(batch_z) #todo: check the dimensiaonlity
      
      
    def forward_set_x(self, set_x, debug=False) -> torch.Tensor:
        """set_x is a datapoint from the perspectives of set-encoding. 
        ie., set_x is a set of datapoints, e.g., a set of (3, 64, 64) images
        
        Args
        ----
        set_x : Iterable[timg]
          of shape (n_imgs_per_bag, nc, h,w), e.g., (10, 3, 64,64)
        
        Returns
        -------
        a set of (flattened) h of each image features, extracted by `self.img_encoder`
        """
        set_h = self.img_encoder(set_x) # (n_imgs_per_bag, last_nf, last_h, last_w); forward function natively opeartes on the batch of inputs, so this operation outputs the batch of image features, as desired
        set_hflat = set_h.view(len(set_h), -1) # (n_imgs_per_bag, self.h_dim)
        assert set_hflat.shape == (self.n_imgs_per_bag, self.h_dim), f"size of set_hflat should be ({self.n_imgs_per_bag, self.h_dim}), but is {set_hflat.shape}"
        
        # poolying-network
        h = self.aggregator(set_hflat)
        z = self.pooling_encoder(h)
        
        # debug
        if debug:
          print("Shapes during forward of a single datapoint (set_x")
          print("set_x.shape (n_imgs_per_bag, nc, h, w): ", set_x.shape)
          print("set_hflat.shape (n_imgs_per_bag, h_dim)", set_hflat.shape)
          print("\tAfter sum aggregator ===")
          print("h.shape (h_dim): ", h.shape)
          print("z.shape (z_dim): ", z.shape)
          
        return z
    
        
        
            
        
        
