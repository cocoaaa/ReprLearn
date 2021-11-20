from .types_ import *
from argparse import ArgumentParser
import torch
import torch.nn as nn

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from ipdb import set_trace as bpt

class ThreeFCs(LightningModule):

    def __init__(self, *,
                 dim_in,
                 nh1,
                 nh2,
                 n_classes):
        """
        Three (fully-collected layer + relu) neural network for img->label classificatino problem

        :param dim_in:
        :param nh1:
        :param nh2:
        :param n_classes:
        """
        super().__init__()
        # Call this to save arguments to __ini__ to the checkpoint file
        self.save_hyperparameters() # sets the property, `self.hparams` as a dictionary of input argument's name and value
        # Equivalent to:
        # self.save_hyperparameters('dim_in', 'nh1','nh2', 'n_classes')
        # Now possible to access `dim_in`, etc from self.hparams


        self.dims = dim_in
        # Define model architecture
        self.layer1 = nn.Linear(dim_in, nh1)
        self.layer2 = nn.Linear(nh1, nh2)
        self.layer3 = nn.Linear(nh2, n_classes)

        # Keep track of the metric(s) to measure the model's performance on
        # this given problem
        self.acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        bs, n_channels, height, width = x.size()

        x = x.view(bs, -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.log_softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        """
        Implements one mini-batch iteration:
         from batch intack -> pass through model -> return loss (ie. computational graph)
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log using tensorboard logger
        # For scalar metrics, self.log will do
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # To log others (eg. images, histograms, graphs, we need to use tb's API directly
        # tensorboard = self.logger.experiment
        # tensorboard = self.logger.experiment
        # tensorboard.add_image()
        # tensorboard.add_histogram(...)
        # tensorboard.add_figure(...)
        # Make a log that will be sent to tensorboard directly
        # log_dict = {'train_acc': acc,
        #        'train_llhd': llhd}
        #

        # Log extra metrics (initiated in __init__)
        # Method 1: Call the tensorboard api (more or less) directly. "more or less"
        # in that we don't need to pass in the `global_step` argument, and
        # if we set on_epoch=True to a kwarg to `self.log`, the accuracies from
        # all iterations will be reduced to a epoch accuracy and will be logged
        # automatically.
        # (1) Functional API for Metrics
        # self.log('train/acc_step', self.acc(logits,y))
        #^ Updates the state of the Metric object by appending new information needed later in .compute

        # Preferred! (2) Or, use Class API
        # First update the Metric object's class, and then pass in the Metric object
        # to self.log -- cf. in (1) we passed in the tensor of accuracy value to
        # write to the event file
        self.acc(logits, y)
        self.log('train/acc', self.acc, on_step=True, on_epoch=False)


        # Alternatively, construct a log_dict and add it to the return statement
        # log_dict = {'train_acc', self.acc(logits, y)}

        return {'loss': loss,
                # 'log': log_dict
                }

    # def training_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('train/acc_epoch', self.acc.compute())


    def validation_step(self, batch, batch_ids):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #Method 1-(1)accumulate metrics on validation batch
        # self.log('val/acc_step', self.val_acc.compute())

        # Preferred. Method 1-(2)
        self.val_acc(logits, y)
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True)

        return {"val_loss": loss}


    # def validation_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('train/acc_epoch', self.val.accu.compute())


    # Second component: Optimization Solver
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []
        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        pass
