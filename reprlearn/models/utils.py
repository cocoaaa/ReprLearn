from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from copy import deepcopy

def make_fc_block(
         in_feats: int,
         out_feats: int,
         act_fn: Optional[Callable]=None,
         use_bn: Optional[bool]=False,
) -> nn.Module:
    act_fn = act_fn or nn.LeakyReLU(0.2, inplace=True)
    return nn.Sequential(
        nn.Linear(in_feats, out_feats),
        nn.BatchNorm1d if use_bn else nn.Identity(),
        act_fn
    )


def inplace_freeze(model: nn.Module):
    """Freezes the modle by turning off its parameter's
    require_grad attributes. In-place operation.
    """
    for p in model.parameters():
        p.requires_grad_(False)


def inplace_unfreeze(model: nn.Module):
    """Freezes the modle by turning off its parameter's
    require_grad attributes. In-place operation.
    """
    for p in model.parameters():
        p.requires_grad_(True)


def compute_ks_for_conv2d(w_in: int, w_out: int, padding: int=1) -> int:
    """Compute the kernel size to use with conv2d when we want
    the output tensor has smaller spatial dimensions, ie.
    w_out < w_in.
    We assume the filter has stride=1.
    Computation is based on the formula
    w_out = floor(w_in - k + 2p) + 1

    We get only positive integer for k only if:
    w_out - w_in < 2p-1
    """
    assert w_out - w_in < 2*padding-1, "No valid kernel size is possible"
    c = w_out - w_in - 2*padding -1
    return -c


def init_zero_grad(params) -> None:
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)


def zero_grad(params) -> None:
    for p in params:
        p.grad = torch.zeros_like(p)


def get_model_ckpt(module) -> Dict[str,torch.Tensor]:
    return {n:deepcopy(p) for n,p in module.named_parameters()}


def get_grad_ckpt(module) -> Dict[str,torch.Tensor]:
    return {n:deepcopy(p.grad) for n, p in module.named_parameters()}


def has_same_values(dict1, dict2) -> bool:
    for v1, v2 in zip(dict1.values(), dict2.values()):
        diff = torch.linalg.norm(v1-v2)
        if not torch.isclose(diff, torch.tensor(0.0)):
            return False
    return True


def model_requires_grad(module) -> bool:
    for p in module.parameters():
        if not p.requires_grad:
            return False
    return True


def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone