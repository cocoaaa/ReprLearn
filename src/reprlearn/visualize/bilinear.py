import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
from data.transforms.transforms import LinearRescaler


def visualize_data_matrix(data: torch.Tensor,
                          n_styles: int, n_contents: int, img_size: Tuple[int],
                          *, title: str = None, normalize: bool = False) -> plt.Figure:
    """
    Visualize 2 or 3dim data matrix

    Args:

    - data (torch.Tensor)
    If data.ndim == 2, data.shape is assumed to be (n_styles*dim_x, n_contents)
    If data.ndim == 3:, data.shape is assumed to be (n_styles, n_contents, dim_x)

    - normalize (bool): project the values of each image by mapping the min and max to 0 and 1
       - Potentialy useful for visualization of gradients or eigenbasis
    """
    dim_x = np.prod(img_size)

    f, ax = plt.subplots(n_styles, n_contents, figsize=(5 * n_contents, 5 * n_styles))
    if title is not None:
        f.suptitle(title)
    f.tight_layout()
    for s in range(n_styles):
        for c in range(n_contents):
            if data.ndim == 2:
                img = data[s * dim_x:(s + 1) * dim_x, c].reshape(img_size)
            elif data.ndim == 3:
                img = data[s, c].reshape(img_size)
            if normalize:
                img = LinearRescaler(target_range=(0,1))(img)
            ax[s][c].imshow(img)
    return f


def visualize_vectors(A: torch.Tensor, is_column: bool = True, title: str = None) -> plt.Figure:
    """
    Visualize each vectors in the input (2dim) tensor as a bar chart

    - A: 2dim tensor whose columns are individual vectors; Assumed to be detached.
    - is_column (bool): if True, assume A to be a collection of column vectors.
        - Otherwise, A is assumed to be a collection of row vectors
    """
    if not is_column:
        A = A.T
    n_vecs = A.shape[1]

    f, ax = plt.subplots(nrows=1, ncols=n_vecs, figsize=(n_vecs * 3, 3))
    if title is not None:
        f.suptitle(title)
    f.tight_layout()
    ax = ax.flatten()
    for i in range(n_vecs):
        vec = A[:, i]
        ax[i].bar(range(len(vec)), vec, label=f'{i}')
        ax[i].set_title(f'Vector {i + 1}')
    return f
