from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from .types_ import *
from .utils import make_fc_block
from IPython.core.debugger import set_trace as breakpoint

class FCDiscriminator(nn.Module):
    """ Discriminator which outputs 1 or 0  for 'real' or 'fake', prediction,
    given 2dim or 3dim input.

    Args
    ----
    in_shape : Tuple[int, int, int] for data_dim==3 or Tuple[int, int] for data_dim=2
    label_map: dictionary mapping 'real', 'gen' labels to numerics (1 or 0).
        Default: {'real':1, 'gen': 0}
    size_average: bool
        If True, average the loss over the data's dimension, in addition to the batch dim


    """
    def __init__(self,
                 in_shape: Union[Tuple[int, int], Tuple[int, int, int]],
                 discr_hidden_dims: List[int] = None,  #Default: [512, 256, 256],
                 label_map: Optional[Dict] = None,
                 act_fn: Optional[Callable] = None,
                 output_type: str='logit', #'logit' or 'prob'
                 size_average: bool = False,
                 ):
        super().__init__()
        self.in_shape = in_shape
        self.data_dim = int(np.prod(in_shape))
        self.discr_hidden_dims = discr_hidden_dims or [512, 256, 256]
        self.label_map = label_map or {"real": 1, "gen": 0}
        self.act_fn = act_fn or nn.LeakyReLU(0.2, inplace=True)
        self.output_type = output_type.lower()
        self.size_average = size_average

        self.n_feats = [self.data_dim, *discr_hidden_dims]
        layers = [make_fc_block(in_, out_, act_fn=self.act_fn) \
                  for (in_, out_) in zip(self.n_feats, self.n_feats[1:])]

        if self.output_type == 'logit':
            self.out_fn = nn.Linear(self.n_feats[-1], 1) #output logit and use nn.BCEwithLogitsLoss as loss fct
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean',
                                                size_average=size_average)
        elif self.output_type == 'prob': # prob(input belongs to class1), aka.
            self.out_fn = nn.Sequential(
                nn.Linear(self.n_feats[-1], 1),
                nn.Sigmoid()
            )
            self.loss_fn = nn.BCELoss(reduction='mean', size_average=size_average)
        else:
            raise ValueError(f"self.output_type must be 'logit' or 'prob'")

        self.model = nn.Sequential(
            *layers,
            self.out_fn
        )

    @property
    def name(self) -> str:
        bn = 'Discr-FC'
        return bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the score for being class 1
        ie. logit, not a probability
        """
        # img_flat = x.view(x.size(0), -1)
        x_flat = x.view(len(x), -1)
        # breakpoint()
        return self.model(x_flat)

    def compute_loss(self, out: torch.Tensor, is_real: bool):
        """Set the proper target_labels depending on whether the preds are based
        on the inputs from the ground-truth (aka. observed dataset) or from the
        generator, and compute the (binary) cross-entropy loss of the predicted
        score (of input is from G.T.) with the proper target_class

        Note: label_map['real'] is the label to indicate the input is from real sample set
            and label_map['gen'] is the label to indicate the input is from generator
        """
        bs = len(out)
        t = self.label_map['real' if is_real else 'gen']  # 1 or 0
        target_label = t * torch.ones_like(out)
        return self.loss_fn(out, target_label)

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser]=None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []

        parser = ArgumentParser(parents=parents, add_help=False)
        parser.add_argument('--discr_hidden_dims', nargs="+", type=int)  # None as default
        parser.add_argument('--discr_type', type=str, default="conv",
                            help='type of layers, conv or resnet (if wanted with skip)')
        parser.add_argument('--discr_act_fn', type=str, default="leaky_relu",
                            help="Act fn for D. Choose relu or leaky_relu (default)")  # todo: proper set up in main function needed via utils.py's get_act_fn function
        parser.add_argument('--discr_out_fn', type=str, default="tanh",
                            help="Output fn for D.  Default tanh")
        return parser


class FCDiscriminator_v1(nn.Module):
    """ Discriminator which outputs 1 or 0  for 'real' or 'fake', prediction,
    given 2dim or 3dim input.

    Args
    ----
    in_shape : Tuple[int, int, int] for data_dim==3 or Tuple[int, int] for data_dim=2
    label_map: dictionary mapping 'real', 'gen' labels to numerics (1 or 0).
        Default: {'real':1, 'gen': 0}
    size_average: bool
        If True, average the loss over the data's dimension, in addition to the batch dim


    """
    def __init__(self,
                 in_shape: Tuple[int, int, int],
                 discr_hidden_dims: List[int] = None,  #Default: [512, 256, 256],
                 label_map: Optional[Dict] = None,
                 act_fn: Optional[Callable] = None,
                 output_type: str='logit', #'logit' or 'prob'
                 size_average: bool = False,
                 ):
        super().__init__()
        self.in_shape = in_shape
        self.data_dim = int(np.prod(in_shape))
        self.discr_hidden_dims = discr_hidden_dims or [512, 256, 256]
        self.label_map = label_map or {"real": 1, "gen": 0}
        self.act_fn = act_fn or nn.LeakyReLU(0.2, inplace=True)
        self.output_type = output_type.lower()
        self.size_average = size_average

        self.n_feats = [self.data_dim, *discr_hidden_dims]
        layers = [make_fc_block(in_, out_, act_fn=self.act_fn) \
                  for (in_, out_) in zip(self.n_feats, self.n_feats[1:])]

        if self.output_type == 'logit':
            self.out_fn = nn.Linear(self.n_feats[-1], 1) #output logit and use nn.BCEwithLogitsLoss as loss fct
        elif self.output_type == 'prob': # prob(input belongs to class1), aka.
            self.out_fn = nn.Sequential(
                nn.Linear(self.n_feats[-1], 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"self.output_type must be 'logit' or 'prob'")

        self.model = nn.Sequential(
            *layers,
            self.out_fn
        )

        #todo: if self.output_type == 'logit':
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean',
                                            size_average=size_average)


    @property
    def name(self) -> str:
        bn = 'Discr-FC'
        return bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the score for being class 1
        ie. logit, not a probability
        """
        # img_flat = x.view(x.size(0), -1)
        x_flat = x.view(len(x), -1)
        # breakpoint()
        return self.model(x_flat)

    def compute_loss(self, score: torch.Tensor, is_real: bool):
        """Set the proper target_labels depending on whether the preds are based
        on the inputs from the ground-truth (aka. observed dataset) or from the
        generator, and compute the (binary) cross-entropy loss of the predicted
        score (of input is from G.T.) with the proper target_class

        Note: label_map['real'] is the label to indicate the input is from real sample set
            and label_map['gen'] is the label to indicate the input is from generator
        """
        bs = len(score)
        t = self.label_map['real' if is_real else 'gen']  # 1 or 0
        target_label = t * torch.ones_like(score)
        return self.loss_fn(score, target_label)

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser]=None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []

        parser = ArgumentParser(parents=parents, add_help=False)
        parser.add_argument('--discr_hidden_dims', nargs="+", type=int)  # None as default
        parser.add_argument('--discr_type', type=str, default="conv",
                            help='type of layers, conv or resnet (if wanted with skip)')
        parser.add_argument('--discr_act_fn', type=str, default="leaky_relu",
                            help="Act fn for D. Choose relu or leaky_relu (default)")  # todo: proper set up in main function needed via utils.py's get_act_fn function
        parser.add_argument('--discr_out_fn', type=str, default="tanh",
                            help="Output fn for D.  Default tanh")
        return parser