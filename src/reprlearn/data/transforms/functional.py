import torch
import numpy as np
from typing import Tuple, Union
from torchvision import transforms as T


def to_3dim(X: torch.Tensor, target_size: Tuple[int, int, int], dtype=torch.float32) -> torch.Tensor:
    """
    Rearragne data matrix X of size (n_styles*dim_x, n_contents)
    to (n_styles, n_contents, dim_x)

    Args:
    - X: torch.Tensor of 2dim data matrix
    - target_size: tuple of n_style, n_contents, dim_x
    """
    assert X.ndim == 2
    n_styles, n_contents, dim_x = target_size
    assert X.shape[0] == n_styles * dim_x
    assert X.shape[1] == n_contents

    target = torch.zeros(target_size, dtype=X.dtype)

    for s in range(n_styles):
        for c in range(n_contents):
            img = X[s * dim_x: (s + 1) * dim_x, c]
            target[s, c] = img
    return target.to(dtype)


def unnormalize(x: Union[torch.Tensor, np.ndarray],
                mean: Union[torch.Tensor, np.ndarray],
                std: Union[torch.Tensor, np.ndarray]):
    """

    :param x: a mini-batch of 3Dim torch.Tensor in order of (bs, c, h, w)
    :param mean: channelwise_mean; (c,)
    :param std:  channelwise_std; (c,)
    :return: a mini-batch of unnormalized 3dim tensors; same shape as input x
    """
    return T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(x)

def to_monochrome(x: torch.Tensor,
                  color: str,
                  preserve_energy:bool=False) -> torch.Tensor:
    """
    Transform a single-channel grayscale (3dim, (1,h,w) tensor)
    to a mono-chrome 3channel tensor (3,h,w), of either gray, red, green, blue.
    - If color is one of [red, green,blue], then outputs a 3-channel tensor by
    putting the input single-channel into the proper channel.
    - If color is gray, then distributes the input grayscale tensor equally into
    the r,g,b channels (ie. puts the input/3 into each r,g,b channels)

    Args:
    x: a single 3dim torch.Tensor a single channel; (1, h, w)
    color: str - one of ['red', 'green', 'blue']

    returns:
    - a single (3dim) torch.Tensor with 3 channels: (3, h, w)

    """
    out = torch.zeros_like(x).repeat((3,1,1))

    color = color.lower()
    color2dim = {"red": 0, "green": 1, "blue": 2}
    if color == 'gray':
        factor = 3.0 if preserve_energy else 1.0
        for i in range(3):
            out[i] = x/factor
    else:
        color_dim = color2dim[color]
        out[color_dim] = x

    return out