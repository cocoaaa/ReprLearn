from pathlib import Path
from reprlearn.utils.misc import change_suffix, get_ith_img_fp, get_ith_npimg
from reprlearn.utils.misc import get_next_version, get_next_version_path 
from reprlearn.utils.misc import info_img_dir

def run_test_change_suffix(fp):
    print(f'old: {fp}')
    print(f"new: {change_suffix(fp, 'txt')}")


def test_change_suffix1():
    fp = Path('/data/hayley/temp.text')
    run_test_change_suffix(fp)


def test_change_suffix2():
    fp = Path('/data/hayley/image.pngng')
    run_test_change_suffix(fp)


def test_get_next_version():
    save_dir = Path('/data/hayley-old/Tenanbaum2000/temp-logs')
    name = 'BiVAE-C_MNIST-M'
    print('next version: ', get_next_version(save_dir, name))


def test_get_next_version_path():
    save_dir = Path('/data/hayley-old/Tenanbaum2000/temp-logs')
    name = 'BiVAE-C_MNIST-M'
    print('next version: ', get_next_version_path(save_dir, name))


def test_get_ith_img_fp():
    sample_dir = Path('/data/hayley-old/Github/GANs/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/output_20220503_run3/celeba_wgan_lp_line/samples_training')
    img0_fp = get_ith_img_fp(img_dir=sample_dir, ind=0)
    print("img0 filepath: ", img0_fp)
    img0 = get_ith_npimg(sample_dir, ind=0,show=True)

    last_img_fp = get_ith_img_fp(img_dir=sample_dir, ind=-1)
    print("last img filepath: ", img0_fp)
    last_img = get_ith_npimg(sample_dir, ind=-1 ,show=True)
    
def test_info_img_dir():
    wang_data_root = Path('/data/datasets/CNN_synth_testset')
    img_dir = wang_data_root / 'cyclegan/apple/0_real' #266 imgs
    # img_dir = wang_data_root / 'crn/0_real' #6382 imgs 

    d = info_img_dir(img_dir)
    assert d['n_imgs'] == 266
    assert len(d['img_sizes']) == 1
    