# src: https://discuss.pytorch.org/t/a-tensorboard-problem-about-use-add-graph-method-for-deeplab-v3-in-torchvision/95808/2
# Use:
# model_wrapper = ModelWrapper(model)
# writer.add_graph(model_wrapper, input_image)
from collections import namedtuple
from typing import Any

import torch

class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data



