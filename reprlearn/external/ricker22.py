from typing import Optional
import numpy as np
from scipy import ndimage
from scipy.fft import dctn
from IPython.core.debugger import set_trace as breakpoint

# Majority of code from ricker22.
# with modifications for esthetics and variable semantics

def _validate_dims(array: np.ndarray) -> None:
    if len(array.shape) != 3:
        raise ValueError(
            f"A batch of single-channel image = should have 3 dimensions (bs, h, w), got {len(array.shape)}."
        )


def log_scale(array: np.ndarray) -> np.ndarray:
    """Take absolute of array and scale logarithmically. Avoid numerical error by adding small epsilon."""
    return np.log(np.abs(array) + 1e-12)


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    """Center crop an image or array of images to square of length size."""
    if size > min(array.shape[-2:]):
        raise ValueError(
            "Image cannot be smaller than crop size (256, 256), got"
            f" {array.shape[-2:]}."
        )

    top = array.shape[-2] // 2 - size // 2
    left = array.shape[-1] // 2 - size // 2
    return array[..., top : top + size, left : left + size]


def dct(gray_imgs: np.ndarray, norm='ortho'):
    """Compute two-dimensional DCT for an array of single channel images."""
    _validate_dims(gray_imgs)

    dct_coeffs = dctn(gray_imgs, axes=(-2, -1), norm=norm)
    return dct_coeffs


def fft(
    gray_imgs: np.ndarray, hp_filter: bool = False, hp_filter_size: int = 3,
    norm: Optional[str]='ortho'
) -> np.ndarray:
    """
    Compute two-dimensional FFT for an array of images, with optional highpass-filtering.
    
    Args:
    -----
    gray_imgs (np.ndarray) : a batch of single-channel images of shape (BS, h, w)
    hp_filter (bool)       : if true, apply high-pass based on median-filter
    hp_filter_size (int)   : kernel size of the median-filter for removing low-frequencies
    """
    _validate_dims(gray_imgs)

    if hp_filter:
        gray_imgs = gray_imgs - ndimage.median_filter(
            gray_imgs, size=(1, hp_filter_size, hp_filter_size)
        )
    print('array shape: ', gray_imgs.shape)
    # breakpoint()
    fft_coeffs = np.fft.fft2(gray_imgs, axes=(-2, -1), norm=norm)
    spectrum = np.fft.fftshift(fft_coeffs, axes=(-2, -1))
    return spectrum


def azimuthal_average(array: np.ndarray) -> np.ndarray:
    """
    Compute the azimuthally averaged radial profile for an array of images.
    Adapted from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """

    def single(image: np.ndarray):
        # Calculate the indices from the image
        y, x = np.indices(image.shape)

        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

        r = np.hypot(x - center[0], y - center[1])

        # Get sorted radii
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]

        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)

        # Find all pixels that fall within each radial bin.
        deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
        rind = np.where(deltar)[0]  # location of changed radius
        nr = rind[1:] - rind[:-1]  # number of radius bin

        # Cumulative sum to figure out sums for each radius bin
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]

        radial_prof = tbin / nr

        return radial_prof

    _validate_dims(array)

    return np.array(list(map(single, array)))


def spectral_density(array: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute one-dimensional power spectrum for an array of images using azimuthal integration.

    :param array: Image array.
    :param normalize: If True, normalize by dividing by DC gain.
    :return: Result array.

    """
    _validate_dims(array)

    spectrum_2d = fft(array)
    spectrum_2d = (
        spectrum_2d.real**2 + spectrum_2d.imag**2
    )  # Schwarz 2021: squared magnitudes
    spectrum_1d = azimuthal_average(spectrum_2d)
    if normalize:
        spectrum_1d = spectrum_1d / spectrum_1d[:, 0][..., np.newaxis]
    return spectrum_1d
