from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from .utils import make_fc_block
from typing import List, Iterable, Callable, Union, Optional, Any, TypeVar, Tuple, Dict


class FCClassifier(nn.Module):

    def __init__(self,
                 #                  in_shape: Tuple[int, int, int],
                 in_feats: int,
                 n_hiddens: Iterable[int],
                 n_classes: Optional[int]=10,
                 act_fn: Optional[Callable]=None,
                 use_bn: Optional[bool]=False,
                 size_average: bool = False,
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.act_fn = act_fn or nn.LeakyReLU(0.2, inplace=True)
        self.act_fn_name = str(act_fn)[3:] # removes nn module's prefix (e.g. `nn.<func-name>`)
        self.use_bn = use_bn

        #         in_feats = int(np.prod(in_shape))
        self.n_feats = [in_feats, *n_hiddens]
        layers = [make_fc_block(in_, out_, self.act_fn, self.use_bn) \
                  for (in_, out_) in zip(self.n_feats, self.n_feats[1:])]

        self.model = nn.Sequential(
            *layers,
            nn.Linear(self.n_feats[-1], self.n_classes) #outputs logit (ie. scores for each class)
        )                                            # we output logit and use nn.BCEwithLogitsLoss
        self.size_average = size_average
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean', size_average=size_average)

        #     nn.Linear(int(np.prod(in_shape)), 512),
        #     self.act_fn,
        #     nn.Linear(512, 256),
        #     self.act_fn,
        #     nn.Linear(256, 1),  # prob(input belongs to class1), aka. "Logit"
        #     #             nn.Sigmoid(), #instead, we output logit and use nn.BCEwithLogitsLoss
        # )

    @property
    def name(self) -> str:
        bn = f'FC_{self.n_layers}-act_{self.act_fn_name}-bn_{self.use_bn}'
        return bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the score for being class 1
        ie. logit, not a probability
        """
        img_flat = x.view(x.size(0), -1)
        return self.model(img_flat)

    def compute_loss(self, score: torch.Tensor, target: torch.Tensor):
        return self.loss_fn(score, target)


def basic_fcnet(in_feats: int, n_layers: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_hiddens = [n_hidden] * n_layers

    return FCClassifier(in_feats=in_feats, n_hiddens=n_hiddens, act_fn=act_fn, use_bn=use_bn)


def fcnet3(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_layers = 2
    return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_bn)


def fcnet5(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_layers = 4
    return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_fn)

def fcnet10(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_layers = 9
    return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_fn)


if __name__ == '__main__':

    image = torch.rand(10, 3, 32, 32)

    # model
    in_feats = int(np.prod(image.shape[1:]))
    n_hidden = 32
    act_fn = None
    use_bn = False

    model = fcnet3(in_feats, n_hidden=n_hidden, act_fn=act_fn, use_bn=use_bn)

    # forward
    output = model(image)
    print(output.shape)
