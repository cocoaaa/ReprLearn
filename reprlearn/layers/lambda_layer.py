# from fastai 
import torch
import torch.nn as nn
from typing import Callable

class Lambda(nn.Module):
    """An easy way to create a pytorch layer for a simple `func` as a nn.Module without
    any learable parameters"""
    
    def __init__(self, func:Callable):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): 
      return self.func(x)


# Instantiated Layers
def Flatten() -> nn.Module:
    "Flattens `x` to a single dimension, often used at the end of a model."
    return Lambda(lambda x: x.view((x.size(0), -1)))
  
  
class Adder(Lambda):
    """Apply torch.sum to the input tensor (e.g., batch of datapts)
    along axis dimension"""
    
    def __init__(self, axis: int, **kwargs):
      super().__init__(lambda x: torch.sum(x, axis=axis, **kwargs))
      
     