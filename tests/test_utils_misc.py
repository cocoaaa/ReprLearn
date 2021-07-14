from pathlib import Path
from src.utils.misc import get_next_version, get_next_version_path


def test_get_next_version():
    save_dir = Path('/data/hayley-old/Tenanbaum2000/temp-logs')
    name = 'BiVAE-C_MNIST-M'
    print('next version: ', get_next_version(save_dir, name))

def test_get_next_version_path():
    save_dir = Path('/data/hayley-old/Tenanbaum2000/temp-logs')
    name = 'BiVAE-C_MNIST-M'
    print('next version: ', get_next_version_path(save_dir, name))