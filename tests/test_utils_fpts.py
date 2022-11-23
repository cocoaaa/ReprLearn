
from copy import deepcopy
from pathlib import Path

import torch
from reprlearn.utils.fpts import estimate_projection
from reprlearn.utils.dists import squared_l2_dist
from reprlearn.utils.misc import read_image_as_tensor, get_ith_img_fp

def test_estimate_projection_1d():
    """test case using 1dim tensor (ie. vectors)
    x_p should be [1.0, 0.0], 
    argmin should be 1.0
    """
    x_g = torch.tensor([1.0, 1.0])
    dset = [torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0])
           ]
    metric = squared_l2_dist
    x_p, d_min = estimate_projection(x_g, dset, metric)

    assert torch.equal(x_p, torch.tensor([1.0, 0.0]))
    assert d_min == 1.0



def test_estimate_projection_3d_self():
    # test1:
    # For the reference manifold: load a bunch of real images used to train this gm, and append the gen. image
    fake_img_dir = Path('/data/datasets/CNN_synth_testset/progan/cow/1_fake/')
    real_img_dir = Path('/data/datasets/CNN_synth_testset/progan/cow/0_real/')
    mani = [read_image_as_tensor(fp) for i, fp in enumerate(real_img_dir.iterdir())
               if i < 10]
    # print(len(mani))
    # show_timgs(mani)

    mani.append(deepcopy(xg))
    # show_timgs(mani)
    # ok good!

    # Load a gen. image whose artifact we want to compute
    xg_fp = get_ith_img_fp(fake_img_dir, 0)
    xg = read_image_as_tensor(xg_fp)
    # show_timg(xg)
    # the closest point (ie. projection) should be the image itself, with dist. 0
    x_p, d_min = estimate_projection(xg, mani) 

    # show result
    # show_timg(x_p, title=d_min)
    # great!


def test_estimate_projection_3d(
    x_g: torch.Tensor,
    manifold_img_dir: Path,
    n_datapts:int = 100,
    n_cols: int = 5,
):
    # test2: For the ref. manifold: do not append the gen image. 
    n_rows = int(np.ceil(n_datapts/n_cols))
    mani = [read_image_as_tensor(fp) for i, fp in enumerate(manifold_img_dir.iterdir())
            if i < n_datapts]
    # print(len(mani))
    # show_timgs(mani, nrows=n_rows, title='images on ref. manifold')


    # Find the closest point (ie. projection) 
    x_p, d_min = estimate_projection(x_g, mani) 

    # show result
    # show_timgs([x_g, x_p], title=f'd_min: {d_min:.2f}', titles=['xg', 'xp'], nrows=1);
    # great!


def test_estimate_projection_3d_on_wang_crn():
    # Test: estimate proj for 3d image tensors
    ## Load a gen. image whose artifact we want to compute
    wang_data_root = Path('/data/datasets/CNN_synth_testset')

    fake_img_dir = wang_data_root / 'crn/1_fake'
    manifold_img_dir =   wang_data_root / 'crn/0_real'

    x_g_fp = get_ith_img_fp(fake_img_dir, 10)
    x_g = read_image_as_tensor(x_g_fp)
    n_datapts = 10

    test_estimate_projection_3d(x_g, manifold_img_dir, 
                            n_datapts=n_datapts)