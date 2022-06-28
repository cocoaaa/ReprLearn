from .types_ import *
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import torch
from torch import Tensor, nn, optim
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from reprlearn.models.utils import inplace_freeze, inplace_unfreeze
from reprlearn.utils.debugger import is_frozen, has_unfrozen_layer # todo: check these
from reprlearn.models.deepset import ImageSetEncoder

# todo: debug
from IPython.core.debugger import set_trace
from reprlearn.visualize.utils import show_timgs

class SetEncoderVectorDecoder(LightningModule):
    """
    A model that 
    - receives an input of a set (e.g., a bag of images), 
    - encodes the *set* to a code vector (z), and 
    - decodes z into the target vector y
    
    Training objective is to minimize the L2 distance between the ground-truth y and 
    the  decoded y (the output from the model).
    
    Args:
    ----
    set_encoder : nn.Module
        A permutation-invariant encoder that operates on a set/sequence of datapts
        set_x --> z
    vec_decoder : nn.Module
        A decoder network that takes the code vector as z and decodes it to a target y_pred
        z --> y_pred
    
    """
    def __init__(self, set_encoder: nn.Module, vec_decoder: nn.Module, loss_fn: Callable,
                 lr: float=0.02):
        super().__init__()
        self.set_encoder = set_encoder
        self.vec_decoder = vec_decoder
        self.loss_fn = loss_fn or nn.MSELoss()
        self.lr = lr 
    
    @property
    def name(self):
        bn = "SetEncoderVecDecoder"
        return bn
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, batch_set_x: Tensor):
        " set_x --> set_encoder --> z --> vec_decoder --> out "
        batch_z = self.set_encoder(batch_set_x)
        out = self.vec_decoder(batch_z)
        return out
    
    
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str,Tensor]: #-> None:
        """Implements one mini-batch iteration:
        Forward: 
          batch input -> pass through model  -> compute loss 
        Backward:
          - zero_grad all learnable parameters' of model (set .grad fields to zeros)
          - optim.step()
          - (optional) update lr scheduler
          - (optional) log current training loss

        Args
        ----
        batch : List or Tuple[Tensor] 
            a batch returned by a dataloader; tuple/list of batch_set_x and batch_set_y
        """
        batch_set_x, batch_y = batch
        # Debug: visualize the batch_set_x 
        # show_timgs(batch_set_x[0].cpu(), title=batch_y[0].cpu())
        # breakpoint()
        
        # forward
        y_pred = self(batch_set_x)
        
        # loss: avg squared-euclidean distance (MSE)
        loss = self.loss_fn(y_pred, batch_y) # what is this ?? --> batch_set_y is a batch of source model id for each set_x
        # optimizer is handled by pl
        
        # log training metric
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    
    def validation_step(self, batch: Dict[str,Any], batch_idx: int) -> None:
        """Evaluate losses on validation dataset (if applicable)

        Args
        ----
        """
        batch_set_x, batch_y = batch
        
        # forward
        y_pred = self(batch_set_x)
        
        # loss: avg squared-euclidean distance (MSE)
        loss = self.loss_fn(y_pred, batch_y) # what is this ?? --> batch_set_y is a batch of source model id for each set_x
        
        # log val metric
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # return loss
    
    def test_step(self, batch: Dict[str,Any], batch_idx: int) -> None:
        """
        Args
        ----
        batch : (Dict[str,Any]) a batch returned by a dataloader
        """
        # let's just make sure
        # print('Debug: Testing...')
        batch_set_x, batch_y = batch
        
        # forward
        y_pred = self(batch_set_x)
        
        # loss: avg squared-euclidean distance (MSE)
        loss = self.loss_fn(y_pred, batch_y) # what is this ?? --> batch_set_y is a batch of source model id for each set_x
        # optimizer
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # return loss
        # -- log each component of the loss
        # return {"test_loss": loss_dict['loss_G']}  # TODO
        
        
        
    @staticmethod  
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []
        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        # parser.add_argument('--in_shape', nargs=3,  type=int, required=True)
        # parser.add_argument('--dec_type', type=str, default="conv",
        #                     help="Generator class type; Default to conv")
        # parser.add_argument('--discr_type', type=str, default="fc",
        #                     help="Discriminator class type. Default to fc (fully-connected)")
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--b1', type=float, default=0.90,
                            help="b1 for Adam optimizer")
        parser.add_argument('--b2', type=float, default=0.90,
                            help="b2 for Adam optimizer")
        parser.add_argument('--act_fn', type=str, default="leaky_relu",
                            help="Choose relu or leaky_relu (default)") #todo: proper set up in main function needed via utils.py's get_act_fn function
        parser.add_argument('--out_fn', type=str, default="tanh",
                          help="Output function applied at the output layer of the decoding process. Default: tanh")
        parser.add_argument('--n_show', type=int, default=64,
                          help="Number of images to log to tensorboard at each ckpt") #todo: remove after implementing this as a callback
        return parser
    
    
class DigitSumPredictor(SetEncoderVectorDecoder):
    """
    Predict the sum of digits in an input set_x (a set of mnist images).
    Input: setx (a seq./Tensor of images from mnist data)
    Output: a scalar value for the sum of digits in setx
    
    Structure:
        - set_encoder: ImageSetEncoder
            $f_enc(setx) = \rho(\sum( \phi(x) for x in setx))$
            where \phi is the instance-level encoder, and $\rho$ is the pooling network,
            and `sum` is used as the aggregator operation.
        - vec_decodoer : 
            Receives a code vector z and decodes into an output vector y_pred
    
    Training objective:
        - Minimize the mean-squared error of (y_pred - y)
        
    """ 
    
    def __init__(self, set_encoder: nn.Module, vec_decoder: nn.Module, lr: float=0.02):
        super().__init__(set_encoder, vec_decoder, loss_fn=nn.MSELoss(), lr=lr)
    
    @property
    def name(self):
        return "DigitSumPredictor"
    
    
class ImageSetEncoder(LightningModule):
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
    def __init__(self, *args, **kwargs):
        """
        Same arguments as for the deepset model as regular Pytorch
        See ImageSetEncoder model in reprlearn/models/deepset.py
        Args
        ----
        in_shape: Iterable[int]
            input of a set with shape (n_imgs_per_bag, nc, h, w)
        z_dim: int
            dimension of the output of the encoder; length of a code vector for each input set
        nfs_img_encoder: Optional[Iterable[int]] = [300, 100, 30]
            number of filters for each conv. layer for the instance-level (image) encoder
        img_encoder_has_bn : bool = True
            True to include batch-norm layers in each conv. block
        img_encoder_act_fn: Callable = nn.LeakyReLU()
            Activation function for each conv. block in the instance-level (image) encoder
        pooling_nfeats: Optional[Iterable[int]] = [100, 30]
            Number of hidden units in the pooling network
        pooling_act_fn: Optional[Callable] = None
            Activation function after each fully-connect layers in the pooling network 
            Default is None, which is replaced with nn.LeakyReLU(0.2, inplace=True) in 
            `make_fc_block` function
        pooling_out_fn: Optional[Callable] = None
            Output function at the end of pooling network, right before outputing the z (code vector)
            Default: nn.Identity()
            
        aggregator_type: Optional[str]='sum', #sum or max
        aggregator_keepdim: bool=False
        
        """
        super().__init__()
        # create imageset model here
        self.set_encoder = ImageSetEncoder(*args, **kwargs)
    
    @property
    def name(self) -> str:
        bn = 'SetEncoder'
        return bn
    
    def forward(self, batch_set_x: Tensor, **kwargs):
        """Delegate forward to the set_encoder"""
        return self.set_encoder(batch_set_x, **kwargs)
    
    def compute_contr1astive_loss(self, batch_z: Tensor, batch_set_y: Tensor):
        """Compute contrastive loss so that z vector in batch_z (batch_z[i]) has a small distancewith the same source id (batch_set__y[i])
        """
        pass
        
    
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str,Tensor]: #-> None:
        """Implements one mini-batch iteration:
        Forward: 
          batch input -> pass through model  -> compute loss 
        Backward:
          - zero_grad all learnable parameters' of model (set .grad fields to zeros)
          - optim.step()
          - (optional) update lr scheduler
          - (optional) log current training loss

        Args
        ----
        batch : List or Tuple[Tensor] 
            a batch returned by a dataloader; tuple/list of batch_set_x and batch_set_y
        """
        batch_set_x, batch_set_y = batch
        # Debug: visualize the batch_set_x 
        show_timgs(batch_set_x[0], title=batch_set_y[0])
        breakpoint()
        
        # forward
        batch_z = self(batch_set_x)
        
        # loss
        # todo: implement contrastive loss here
        loss = self.compute_contrastive_loss(batch_z, batch_set_y) #batch_set_y is a batch of source model id for each set_x
        # optimizer
        
        # log training metric
        
    
    def validation_step(self, batch: Dict[str,Any], batch_idx: int) -> None:
        """Evaluate losses on validation dataset (if applicable)

        Args
        ----
        """
        pass
    
    def test_step(self, batch: Dict[str,Any], batch_idx: int) -> None:
        """
        Args
        ----
        batch : (Dict[str,Any]) a batch returned by a dataloader
        """
        # with torch.no_grad(): -- handled by pl
        # let's just make sure
        # print('Debug: Testing...')
        # print('\tis G in train mode: ', self.generator.training)
        # print('\tis D in train mode: ', self.generator.training)

        loss_dict = self.push_through_D_and_G(batch)

        # -- log each component of the loss
        self.log('test/loss_G', loss_dict['loss_G'])
        self.log('test/loss_D', loss_dict['loss_D'])
        self.log('test/loss_D_gen', loss_dict['loss_D_gen'])
        self.log('test/loss_D_real', loss_dict['loss_D_real'])

        # return {"test_loss": loss_dict['loss_G']}  # TODO
        
        
        #todo: clean up and add new arguments if needed
        
    @staticmethod  
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []
        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--b1', type=float, default=0.90,
                            help="b1 for Adam optimizer")
        parser.add_argument('--b2', type=float, default=0.90,
                            help="b2 for Adam optimizer")
        parser.add_argument('--act_fn', type=str, default="leaky_relu",
                            help="Choose relu or leaky_relu (default)") #todo: proper set up in main function needed via utils.py's get_act_fn function
        parser.add_argument('--out_fn', type=str, default="tanh",
                          help="Output function applied at the output layer of the decoding process. Default: tanh")
        parser.add_argument('--n_show', type=int, default=64,
                          help="Number of images to log to tensorboard at each ckpt") #todo: remove after implementing this as a callback
        return parser


