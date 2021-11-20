from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from .types_ import *

class FCDiscriminator(nn.Module):

    def __init__(self,
                 in_shape: Tuple[int, int, int],
                 label_map: Optional[Dict] = None,
                 size_average: bool = False,
                 ):
        super().__init__()
        self.label_map = label_map or {"real": 1, "gen": 0}
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(in_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # prob(input belongs to class1), aka. "Logit"
            #             nn.Sigmoid(), #instead, we output logit and use nn.BCEwithLogitsLoss
        )
        self.size_average = size_average
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
        img_flat = x.view(x.size(0), -1)
        return self.model(img_flat)

    def compute_loss(self, score: torch.Tensor, is_real: bool):
        """Set the proper target_labels depending on whether the preds are based
        on the inputs from the ground-truth (aka. observed dataset) or from the
        generator, and compute the (binary) cross-entropy loss of the predicted
        score (of input is from G.T.) with the proper target_class

        Note: label_map['real'] is the label to indicate the input is from real sample set
            and label_map['gen'] is the label to indicate the input is from generator
        """
        bs = len(score)
        t = self.label_map['real' if is_real else 'gen']  # 0 or 1
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