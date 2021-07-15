import os,sys
import joblib

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.linalg import norm as tnorm

from src.visualize.bilinear import visualize_data_matrix, visualize_vectors

"""
TODO:
- [ ] Test w/ nonlinear function
    - eg: Sigmoid because the target tensor of images will be scaled to [0,1]

"""
class SymBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        # Cache to store previous iteration's values (Eg. parameters)
        self.cache = {}

    def cache_params(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                self.cache[name] = param.detach().clone()

    def some_params_not_changed(self) -> bool:
        """
        Compares if each of the current named parameter has changed from the values
        stored in self.cache.
        - Assumes self.cache is a dictionary that stores this model's named parameters
        - Useful to check if parameters are being modified across different iterations

        """
        with torch.no_grad():
            not_changed = {}
            for name, param in self.named_parameters():
                if torch.equal(self.cache[name], param):
                    d = self.cache[name] - param
                    not_changed[name] = torch.linalg.norm(d)
                    print(tnorm(self.cache[name]), tnorm(param))
                    # print(tnorm(param.grad))
            if len(not_changed) < 1:
                return False
            else:
                print(f"Not changed: \n", not_changed)
                return True

    def all_params_changed(self) -> bool:
        return not self.some_params_not_changed()



class ShallowSymBilinear(SymBilinear):
    """
    Two-factor model implemented as a single tensor multiplication followed by
    a non-linear function.

    Init args:
    - n_styles: number of styles to learn
    - n_contents: number of contents to learn
    - dim_style: dimensionality of each style vector
    - dim_content: dimensiaonlity of each content vector
    - dim_x: dimensionality of each data variable
    - non_linear_fn: a torch.nn.functional object that is applied after the bilinear tensor multiplication
        applied to a pair of content and style vectors

    Model parameters:
    - styles: a tensor containing all style vectors;
        - of shape (n_styles, 1, 1, dim_style)
    - contents: a tensor containing all content vectors;
        - of shape (n_contents, 1, 1, 1, dim_content)
    - W: a 3dim tensor that contains pairwise weights for each style dimension i and content dimension j
    in constructing kth dimension of a data vector;
        - of shape (dim_datapoint, dim_style, dim_content)

    """
    def __init__(self, *,
                 n_styles: int, n_contents: int,
                 dim_style: int, dim_content: int, dim_x: int,
                 n_layers: int = 1, non_linear_fn: Callable = nn.Identity(),
                 dtype=torch.float32, device='cpu'):
        super().__init__()  # just the nn.Module
        self.n_styles = n_styles
        self.n_contents = n_contents
        self.I, self.J, self.K = dim_style, dim_content, dim_x
        self.non_linear_fn = non_linear_fn
        self.dtype = dtype
        self.device = device

        # Model parameters
        self.styles = nn.Parameter(
            torch.randn((n_styles, 1, 1, dim_style), dtype=dtype)
        )  # A: each row is a style vector; (S,1,1,I)
        self.contents = nn.Parameter(
            torch.randn((n_contents, 1, 1, dim_content, 1), dtype=dtype)
        )  # B: each column is a content vector; (C, 1,1,J, 1)
        self.W = nn.Parameter(
            torch.randn((dim_x, dim_style, dim_content), dtype=dtype)
        )  # W: (K,I,J)


    def forward(self, *, s: int = None, c: int = None):
        """
        s: style label; must be in {0,...,n_styles-1}
        c: content label; must be in {0,..., n_contents-1}
        """
        #         assert self.styles[s].shape == (1,self.I)
        #         assert self.contents[c].shape == (1,self.J)
        A = self.styles
        B = self.contents
        if s is not None:
            A = self.styles[[s]]
        if c is not None:
            B = self.contents[[c]]
        out = A.matmul(self.W)
        #         print(out.shape)
        out = out.matmul(B)
        #         print(out.shape) #(C,S,K,1,1)

        # Following the convention of fig 3,4 in Tenanbaum2000's ,
        # we output the reconstructed datapoints as a tensor of size (S,C,K)
        out = self.non_linear_fn(out.permute(1, 0, 2, -2, -1).squeeze())  # (S,C,K)

        return out

    def shortname(self):
        return f"shallow-bilinear-{self.non_linear_fn}"

    def descr(self):
        return f"{self.shortname()}_S:{self.n_styles}_I:{self.I}_C:{self.n_contents}_J:{self.J}_K:{self.K}"

    def show_params(self, img_size: Tuple[int]):
        with torch.no_grad():
            visualize_vectors(self.styles.squeeze(), is_column=False, title='Styles');
            visualize_vectors(self.contents.squeeze(), is_column=False, title='Contents');
            visualize_data_matrix(self.W.permute(1, 2, 0), self.I, self.J,
                                  img_size, title='W', normalize=True);

    def show_grads(self, img_size: Tuple[int]):
        with torch.no_grad():
            visualize_vectors(self.styles.grad.squeeze(), is_column=False, title='Styles');
            visualize_vectors(self.contents.grad.squeeze(), is_column=False, title='Contents');
            visualize_data_matrix(self.W.grad.permute(1, 2, 0), self.I, self.J,
                                  img_size, title='W', normalize=True);


# class DeepSymBilinear(SymBilinear):
#         """
#         Two-factor model implemented as a stack of non-linear functions via DNN.
#
#         Init args:
#         - n_styles: number of styles to learn
#         - n_contents: number of contents to learn
#         - dim_style: dimensionality of each style vector
#         - dim_content: dimensiaonlity of each content vector
#         - dim_x: dimensionality of each data variable
#         - n_layers: number of layers in a function that maps a content vector and a style vector to
#             a datapoint of that content and style.
#         - non_linear: a torch.nn.functional object that is applied after the bilinear tensor multiplication
#             applied to a pair of content and style vectors
#
#
#         Model parameters:
#         - styles: a tensor containing all style vectors;
#             - of shape (n_styles, 1, 1, dim_style)
#         - contents: a tensor containing all content vectors;
#             - of shape (n_contents, 1, 1, 1, dim_content)
#         - W: a 3dim tensor that contains pairwise weights for each style dimension i and content dimension j
#         in constructing kth dimension of a data vector;
#             - of shape (dim_datapoint, dim_style, dim_content)
#
#         """
#         def __init__(self, *,
#                      n_styles: int, n_contents: int,
#                      dim_style: int, dim_content: int, dim_x: int,
#                      n_layers: int = 1, non_linear_fn: Callable = nn.Identity(),
#                      dtype=torch.float32, device='cpu'):
#             super().__init__()  # just the nn.Module
#             self.n_styles = n_styles
#             self.n_contents = n_contents
#             self.I, self.J, self.K = dim_style, dim_content, dim_x
#             self.n_layers = n_layers
#             self.non_linear_fn = non_linear_fn
#             self.dtype = dtype
#             self.device = device
#
#             # Model parameters
#             self.styles = nn.Parameter(
#                 torch.randn((n_styles, 1, 1, dim_style), dtype=dtype)
#             )  # A: each row is a style vector; (S,1,1,I)
#             self.contents = nn.Parameter(
#                 torch.randn((n_contents, 1, 1, dim_content, 1), dtype=dtype)
#             )  # B: each column is a content vector; (C, 1,1,J, 1)
#             self.Ws = torch.randn((n_layers, dim_x, dim_style, dim_content),
#                                   dtype=dtype)                           )  # (n_layers, K,I,J)
#             for i in range(n_layers):
#                 self.Ws[i] = torch.randn((dim_x, dim_style, dim_content))
#             self.Ws = nn.Parameter(self.Ws)
#
#         def forward(self, *, s: int = None, c: int = None):
#             """
#             s: style label; must be in {0,...,n_styles-1}
#             c: content label; must be in {0,..., n_contents-1}
#             """
#             #         assert self.styles[s].shape == (1,self.I)
#             #         assert self.contents[c].shape == (1,self.J)
#             A = self.styles
#             B = self.contents
#             if s is not None:
#                 A = self.styles[[s]]
#             if c is not None:
#                 B = self.contents[[c]]
#             out = A.matmul(self.W)
#             #         print(out.shape)
#             out = out.matmul(B)
#             #         print(out.shape) #(C,S,K,1,1)
#
#             # Following the convention of fig 3,4 in Tenanbaum2000's ,
#             # we output the reconstructed datapoints as a tensor of size (S,C,K)
#             out = self.non_linear_fn(out.permute(1, 0, 2, -2, -1).squeeze())  # (S,C,K)
#
#             # todo: more layers
#             return out
#
#         def shortname(self):
#             return f"{self.n_layers}layer-bilinear-{self.non_linear_fn}"
#
#         def descr(self):
#             return f"{self.shortname()}-S:{self.n_styles}-I:{self.I}-C:{self.n_contents}_J:{self.J}_K:{self.K}"
#
#         def show_params(self, img_size: Tuple[int]):
#             with torch.no_grad():
#                 visualize_vectors(self.styles.squeeze(), is_column=False, title='Styles');
#                 visualize_vectors(self.contents.squeeze(), is_column=False, title='Contents');
#                 visualize_data_matrix(self.W.permute(1, 2, 0), self.I, self.J,
#                                       img_size, title='W', normalize=True);
#
#         def show_grads(self, img_size: Tuple[int]):
#             with torch.no_grad():
#                 visualize_vectors(self.styles.grad.squeeze(), is_column=False, title='Styles');
#                 visualize_vectors(self.contents.grad.squeeze(), is_column=False, title='Contents');
#                 visualize_data_matrix(self.W.grad.permute(1, 2, 0), self.I, self.J,
#                                       img_size, title='W', normalize=True);


class ResidualBilinear(nn.Module):
    pass


class TwoLayerResidualBilinear(ResidualBilinear):
    pass


class MultiLayerResidualBilinear(ResidualBilinear):
    pass