from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Iterable, Callable, Union, Optional, Any, TypeVar, Tuple, Dict


class CNN3Classifier(nn.Module):

    def __init__(self, nf1: int, nf2: int, nf3:int, 
                 fc_nf1: int, fc_nf2: int, n_classes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, nf1, kernel_size=5, padding=1)  
        self.conv2 = nn.Conv2d(nf1, nf2, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(nf2, nf3, kernel_size=5, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2) 

        self.fc1 = nn.Linear(6 * 6 * nf3, fc_nf1)
        self.fc2 = nn.Linear(fc_nf1, fc_nf2)
        self.fc3 = nn.Linear(fc_nf2, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # (bs, nf1, 64/2,64/2)
        x = self.pool(F.relu(self.conv2(x))) # (bs, nf2, 64/4,64/4)
        x = self.pool(F.relu(self.conv2(x))) # (bs, nf3, 64/8,64/8) 
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    def encode(self,x):
        """Returns the pan-ultimate layer as the code vector of an input.
        
        Args:
        -----
        x : batch of inputs (N, nC, h=64, w=64)
        Returns:
        z : batch of code vectors returned at the pen-ultimate layer
            of size (N, self.fc_nf2)
        """
        x = self.pool(F.relu(self.conv1(x))) # (bs, nf1, 64/2,64/2)
        x = self.pool(F.relu(self.conv2(x))) # (bs, nf2, 64/4,64/4)
        x = self.pool(F.relu(self.conv2(x))) # (bs, nf3, 64/8,64/8) 
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        
        return x
     
    @property
    def name(self) -> str:
        return f'CNN3-FC3'



# def basic_fcnet(in_feats: int, n_layers: int, n_hidden: int, act_fn: Callable, use_bn: bool):
#     n_hiddens = [n_hidden] * n_layers

#     return FCClassifier(in_feats=in_feats, n_hiddens=n_hiddens, act_fn=act_fn, use_bn=use_bn)


# def fcnet3(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
#     n_layers = 2
#     return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_bn)


# def fcnet5(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
#     n_layers = 4
#     return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_fn)

# def fcnet10(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
#     n_layers = 9
#     return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_fn)


if __name__ == '__main__':

    bs = 4
    batch_x = torch.rand(bs, 3, 64, 64)

    # model parameters
    model_params = {
        'nf1': 100,
        'nf2': 100,
        'nf3': 100,
        'fc_nf1': 120,
        'fc_nf2': 50,
        'n_classes': 10
    }

    
    model = CNN3Classifier(**model_params)

    # forward
    output = model(batch_x)
    print('output size: ', output.shape) 
    assert(output.shape == (bs, model_params['n_classes']) )
