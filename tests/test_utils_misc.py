from pathlib import Path
from reprlearn.utils.misc import change_suffix
from reprlearn.utils.misc import get_next_version, get_next_version_path


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