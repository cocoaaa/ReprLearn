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