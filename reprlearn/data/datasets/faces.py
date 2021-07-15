import torch
from typing import Tuple

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