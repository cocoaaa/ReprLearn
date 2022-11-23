from colorsys import rgb_to_hls
import PIL
import numpy as np
from scipy import ndimage
from typing import Any,List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io
from pathlib import Path
from reprlearn.visualize.utils import show_npimgs
from IPython.core.debugger import set_trace as breakpoint


def crop_center(img: np.ndarray, target_h: int, target_w: int):
    y,x = img.shape[:2]
    startx = x//2-(target_w//2)
    starty = y//2-(target_h//2)    
    return img[starty:starty+target_h,startx:startx+target_w]

#############################################################################################
# Compute magnitude of spectrum of image in grayscale
#
#
#############################################################################################
def compute_magnitude_spectrum_of_grayscale_np(img: np.ndarray,
                               target_size: Tuple[int,int]=None,
                               show: bool=False):
    """Read the image from `img_fp` as a float image in grayscale, i.e. in range [0.0., 1.0]),
    and transform the image from the spatial domain to frequency domain via np.fft.fft2, 
    and compute the magnitude spectrum using np.abs on the frequency representation
    return this magnitude spectrum.
    
    Note that since the DC component (the energy at frequency (u,v=0,0) 
    is very large compared to energy in other frequenceis (u,v)'s,
    to visualize this magnitude spectrum, show in the logscale, e.g., plt.imshow(np.log(mag_spectrum))
    
    Args
    ----
    img_fp : Path
    target_size : Tupel[int, int]
        (target_h, target_w). if None, use the height and width of the image.
        
    Returns
    -------
        magnitude_spectrum: a 2dim numpy array, same sized as the input img
        
    """
    assert len(img.shape) == 2, "This function works on only grayscale image"
    if target_size is None:
        target_size = img.shape
        
    # Take the 2-dimensional DFT and centre the frequencies
    f = np.fft.fft2(img, s=target_size)
    fshift = np.fft.fftshift(f) 
    mag_spectrum = np.abs(fshift)
    
    if show:
        plt.imshow(np.log(mag_spectrum))
        plt.title('Log(magnitude spectrum)')
        
    return mag_spectrum


def compute_magnitude_spectrum_of_grayscale_fp(img_fp: Path, 
                               target_size: Optional[Tuple[int,int]]=None,
                               transform: Optional[Callable]=None,
                               show: bool=False):
    """Compute the mag. of spectrum of an image in `img_fp` as a grayscale
    Input: path to an image (could be rgb)
    Ouput: magnitude of spectrum F of the image in grayscale
    
    Args:
    ----
    img_fp : Path to the image file
    target_size : (target_h, target_w) of the output 2D dft'ed spsectrum
    transform: Callable
        apply this function on npimage if not None
    show : true to show the mag. spectrum in logscale
    
    Returns
    -------
    mag_spectrun of the image in grayscale
        
    """
    img = io.imread(img_fp, as_gray=True) #float64 image in grayscale (so no color channel); [0.0., 1.0]
    
    if transform is not None:
        img = transform(img)

    mag_spectrum = compute_magnitude_spectrum_of_grayscale_np(img, target_size, show)
    return mag_spectrum


def compute_onechannel_ffts(img_fp: Path,
                            kernel_size: int, 
                            use_grayscale:bool=True,
                            channel_ind: Optional[int]=None,
                            show: bool=False,
                           ) -> List[np.array]:
    """ Given img in img_fp, 
    either convert rgb to grayscale )if `use_grayscale` is true,
    or take the `channel_ind`th channel of the image if `use_grayscale` is false 
    and `channel_ind` is not None,
    
    Returns: a list of 2d spectrums of img, blurred img, highpass img, and 
    highpass image whose values are clipped to range [0,1]
    """
    # image domain
    if use_grayscale:
        img = io.imread(img_fp, as_gray=True)
    elif channel_ind is not None:
        img = io.imread(img_fp, as_gray=False)[:,:,channel_ind]
    else:
        print( "Must either use_grayscale or specify the channel index")
        raise ValueError
        
    blurred_img = grayscale_median_filter(img, kernel_size=kernel_size)
    high_img = img - blurred_img
    clipped_high_img = clip_floatimg(high_img)

    # freq. domain
    f = compute_magnitude_spectrum_of_grayscale_np(img)
    low_f = compute_magnitude_spectrum_of_grayscale_np(blurred_img)
    high_f = compute_magnitude_spectrum_of_grayscale_np(high_img)
    clipped_high_f = compute_magnitude_spectrum_of_grayscale_np(clipped_high_img)

    if show:
        show_npimgs(
            [np.log(f), np.log(low_f), np.log(high_f), np.log(clipped_high_f)], 
            titles=['f', 'low_f', 'high_f', 'clipped_high_f'], 
            nrows=2
        );
    return {"f": f,
            "low_f": low_f,
            "high_f": high_f,
            "clipped_high_f": clipped_high_f
           }
        

    



def compute_avg_magnitude_spectrum_of_grayscale(img_dir: Path, 
                                   target_size: Optional[Tuple[int, int]]=None,
                                   max_n_imgs: Optional[int]=2_000,
                                   transform: Optional[Callable]=None,
                                   verbose: bool=False,
                                  ):
    """Compute Avg. mag. spectrum using (all or max_num_samples) images in the img_dir folder.
    If the image is rgb, we first convert it to grayscale before applying the DFT.
    
    """
    avg_mag_spectrum = None
    
    n_imgs = 0 
    for img_fp in img_dir.iterdir():
        if not img_fp.is_file():
            continue
        mag_spectrum = compute_magnitude_spectrum_of_grayscale_fp(img_fp, target_size=target_size, transform=transform)

        if avg_mag_spectrum is None:
            avg_mag_spectrum = mag_spectrum
        else:
            try:  # some images in this img_dir may have different shape. if so, we skip until finding the same shape
                avg_mag_spectrum += mag_spectrum
            except ValueError as e:
                if verbose:
                    print("ValueError occured. Not adding this image to average")
                continue

        # increment the count of images used for averaging
        n_imgs += 1
        if n_imgs == max_n_imgs:
            break
        
    avg_mag_spectrum /= n_imgs
    return avg_mag_spectrum             
                        




#############################################################################################
# Our implementation of Zhang et al's input pre-processing.
# Computes magnitude of spectrum of image in RGB: apply DFT to each channel and stack them as F_rgb
#
#############################################################################################
def compute_magnitude_spectrum_channelwise_np(img: np.ndarray,
                                           target_size: Optional[Tuple[int, int]]=None,
                                           ):
    """ 
    img : np.ndarray
        a rgb image in float (range [0.0, 1.0])
    """
    assert img.shape[-1] in [3,4], f"this function works only on 3- or 4-channel image: {img.shape[-1]}"
    if img.shape[-1] == 4:
        img = img[:,:,:3] #remove alpha channel
        # print('Removed alpha channel...') # debug


        
    if target_size is None:
        target_size = img.shape[:2]
    id2color = {0: 'r', 1:'g', 2:'b'}
    
    mag_f_rgb = np.zeros((*target_size, 3))
    for id_c, color_name in id2color.items():
        channel = img[:,:,id_c]
        mag_channel = compute_magnitude_spectrum_of_grayscale_np(channel, target_size)
        mag_f_rgb[:,:,id_c] = mag_channel
        
    return mag_f_rgb


def compute_magnitude_spectrum_channelwise_fp(
    img_fp: Path, 
    target_size: Optional[Tuple[int, int]]=None,
    transform: Optional[Callable]=None,
    ):
    
    img = io.imread(img_fp) / 255.0 # float image [0.0, 1.0]
    if transform is not None:
        img = transform(img)
    #todo: clip?

    mag_f_rgb = compute_magnitude_spectrum_channelwise_np(img, target_size)
        
    return mag_f_rgb


def compute_normed_log_mag_spectrum_channelwise_fp(img_fp: Path,
                                        target_size: Tuple[int, int],
                                        normalize: bool=True,
                                        show: bool=False,
                                        transform: Optional[Callable]=None,
                                       ):
    """My implementation of the pre-proc in Zhang2020 for input to spectral classifier (A_classifier).
    - compute spectrum of each channel in the rgb image
    - collet the channel's spectrums as each ouchannel of np array: `f_rgb`
    - compute the mag. of the spectrum: `mag_f_rgb = np.abs(f_rgb)`
    - compute log of this 3cannel spectrum output array: `log_mag_f_rgb`
    - normalize the log_mag_F_rgb (of 3 channels) to [-1,1]
    
    The resulting 3channel input (log_mag_f_rgb) is used as an input to train their classifier.
    
    """
    mag_rgb = compute_magnitude_spectrum_channelwise_fp(img_fp, target_size, transform)
#     info(mag_rgb) #0.09 ~ 23299

    #log mag
    log_mag_rgb = np.log(mag_rgb) 
#     info(log_mag_rgb) #-2.4 ~ 10.

    if normalize:
        # normalize to [-1,1]
        normed_log_mag = normalize_to_negposone(log_mag_rgb)

    if show:
        plt.imshow(normed_log_mag)
        plt.title('log magnitutude of channelwise spectrum, normed to [-1,1]')
    return normed_log_mag


def compute_avg_normed_log_mag_spectrum_channelwise(
    img_dir: Path, 
    target_size: Tuple[int, int],
    normalize: Optional[bool]=True,
    max_n_imgs: Optional[int]=2_000,
    transform: Optional[Callable]=None,

):
    """Take the average of normed_log_mag_f_rgb of all images in the img_dir.
    Each image undergoes the pre-processing used to train A-classifier in zhang2020.
    - Apply DFT to each channel in img_rgb:  `f_rgb`
    - Get magnitude of f_rgb: `mag_f_rgb = np.abs(f_rgb)`
    - Take log and normalize it to range [-1,1]: `normed_log_mag_f_rgb`
    
    Returns the average `normed_log_mag_f_rgb` over the image dir.
    So, the returned 3channel np.array shows the average input to the zhang2020's classifier.
    """
    avg_mag_spectrum = None
    n_imgs = 0 
    for img_fp in img_dir.iterdir():
        if not img_fp.is_file():
            continue
        mag_spectrum = compute_normed_log_mag_spectrum_channelwise_fp(
            img_fp, 
            target_size=target_size, 
            normalize=normalize,
            transform=transform
            )

        if avg_mag_spectrum is None:
            avg_mag_spectrum = mag_spectrum
        else:
            avg_mag_spectrum += mag_spectrum
            
            
        n_imgs += 1
        if n_imgs == max_n_imgs:
            break
        
    avg_mag_spectrum /= n_imgs
    return avg_mag_spectrum
    





#############################################################################################
# Our implementation of Wang2020 et al's average spectra as in Fig7.
# todo: check if it actually is wang2020's
# 20221019-153742 update: try out the code from AutoGAN paper: https://tinyurl.com/23dmhoym
#############################################################################################
def compute_log_mag_fft(img_fp: Path, 
                        as_gray:bool=False,
                        transform:Callable=None,
                        mode:Optional[int]=None):
    """20221019-153742: 
    try out the code from AutoGAN paper: https://tinyurl.com/23dmhoym
    
    Workflow:
    for each channel:
        - compute fft
        - convert to log scale for each values (for better viz)
        - normalize the log-fft image to -1,1
        - clip values <-1 to -1 and values >1 to 1
    return log-fft
    """
    img = io.imread(img_fp, as_gray=as_gray) /255. #float64 image in grayscale (so no color channel); [0.0., 1.0]
    if as_gray:
        img = img[:,:,None]
    nc = img.shape[-1]
    if transform is not None:
        img = transform(img) 
    #channelwise log-fft
    # (h,w,nc)
    for i in range(nc):
        channel = img[:,:,i]
        fft_img = np.fft.fft2(channel)
        fft_img = np.log(np.abs(fft_img) + 1e-3)
        fft_min = np.percentile(fft_img, 5)
        fft_max = np.percentile(fft_img, 95)
        fft_img = (fft_img - fft_min) / (fft_max - fft_min)
        fft_img = (fft_img - 0.5) * 2
        fft_img[fft_img < -1] = -1
        fft_img[fft_img > 1] = 1
        # if mode is not None:
        #     fft_img = np.fft.fftshift(fft_img)
        #     # set low freq to 0
        #     if mode == 0:
        #         fft_img[:21, :] = 0
        #         fft_img[:, :21] = 0 
        #     #set mid and high freq to 0
        #     if mode == 1: 
        #         fft_img[:57, :] = 0
        #         fft_img[:, :57] = 0
        #         fft_img[177:, :] = 0
        #         fft_img[:, 177:] = 0
        #     #set low and high freq to 0
        #     elif mode == 2:
        #         fft_img[:21, :] = 0
        #         fft_img[:, :21] = 0
        #         fft_img[203:, :] = 0
        #         fft_img[:, 203:] = 0
        #         fft_img[57:177, 57:177] = 0
        #     #set low and mid freq to 0
        #     elif mode == 3:
        #         fft_img[21:203, 21:203] = 0
        #     fft_img = np.fft.fftshift(fft_img)
        img[:,:,i] =fft_img
    return img
    
    
    
def compute_avg_mag_spectrum_channelwise(
    img_dir: Path, 
    target_size: Optional[Tuple[int, int]]=None,
    max_n_imgs: Optional[int]=2_000,
    transform: Optional[Callable]=None,
    verbose: bool=False
):
    """ver0
    Take the average of normed_log_mag_f_rgb of all images in the img_dir.
    Each image undergoes the pre-processing used to train A-classifier in zhang2020.
    - Apply DFT to each channel in img_rgb:  `f_rgb`
    - Get magnitude of f_rgb: `mag_f_rgb = np.abs(f_rgb)`
    
    Returns the average `normed_log_mag_f_rgb` over the image dir.
    So, the returned 3channel np.array shows the average input to the zhang2020's classifier.
    
    Args
    ----
    transform: Optional[Callable]
        a function that is applied to each image right aftet it is read from img_dir,
        before channelwise DFT's are computed.
    Returns
    --------


    
    """
    avg_f_rgb = None
    n_imgs = 0 
    for img_fp in img_dir.iterdir():
        if not img_fp.is_file():
            continue
        
        #todo: apply high-pass filtering in pixel domain
        #by applying a media-blurring (in pixels)
        img = io.imread(img_fp) / 255.0 # float image [0.0, 1.0]
        # here
        if transform is not None:
            img = transform(img)

        #compute f_rgb of this image: 
        mag_f_rgb = compute_magnitude_spectrum_channelwise_np(img, target_size)

        if avg_f_rgb is None:
            avg_f_rgb =  mag_f_rgb
        else:
            try:  # some images in this img_dir may have different shape. if so, we skip until finding the same shape
                avg_f_rgb += mag_f_rgb
            except ValueError as e:
                if verbose:
                    print("ValueError occured. Not adding this image to average")
                continue

        # increment the count of images used for averaging
        n_imgs += 1
        if n_imgs == max_n_imgs:
            break    
        
    avg_f_rgb /= n_imgs
    return avg_f_rgb
    



# spatial filters
def grayscale_median_filter(img: np.ndarray, kernel_size):
    return ndimage.median_filter(img, size=kernel_size)

def channelwise_median_filter(img: np.ndarray, kernel_size):
#     assert (len(img.shape) == 3), "this function supports 3dim np-image"
#     assert (img.shape[-1] == 3), "this function support only 3channel images"
    
    out = np.zeros_like(img)
    nc = img.shape[-1]
    for i in range(nc):
        channel = img[:,:,i]
        out[:,:,i] = ndimage.median_filter(channel,size=kernel_size)
        
    return out
    
    
def clip_floatimg(img: np.ndarray, cmin: float=0.0, cmax: float=1.0):
    return np.clip(img, cmin, cmax)    
    
    
    
    
    
# cropping
def get_crop_tl(img_size: Tuple[int,int], #h,w
                crop_roi_size: Tuple[int,int], #h,w
               ) -> Tuple[int,int]:
    """Compute the top-left array index to the crop ROI of the original image `img`
    which ROI's h,w to `target_h` and `target_w`.
    
    
         x     tx  hx
    (0,0)-->---|---|-----------|
    y |                        |
      \/      hcrop_w          |
 ty-->|        |---|---|       |
      |  hcrop_h-  .   |       |
      |        |-------|       |       
      |                        |
      |                        |
      |------------------------|
    
    - Use to get the coordinate of the mask of crop ROI
    
    Returns
    -------
    - (tx, ty) : Tuple[int, int]
    
    """
    img_h, img_w = img_size
    target_h, target_w = crop_roi_size
    if target_h >= img_h:
        target_h = img_h
    if target_w >= img_w:
        target_w = img_w
        
    hcrop_w = target_w//2 # half of the width of cropped region
    hcrop_h = target_h//2# half of the height of cropped region
    
    return img_h//2 - hcrop_h, img_w//2 - hcrop_w
    
    
def get_center_mask(
    img_size: Tuple[int, int], #h,w
    mask_size: Tuple[int, int], #h,w
    fill_value: bool=True, #todo: can generalize it to float numbers
) -> np.ndarray:
    """Get the mask np array as the same shape as img_size
    where the center ROI of size `mask_size` is filled with value `fill_value`,
    and cells outside of the ROI are filled with zero's.
    
    Returns
    -------
    np.ndarray of img_shape
    
    """
    outside_value = not fill_value
    mask = np.zeros((img_size))
    roi_h, roi_w = mask_size
    mask.fill(outside_value)
    tx, ty = get_crop_tl(img_size, mask_size)

    hmask_w, hmask_h = mask_size[0]//2, mask_size[1]//2


    mask[ty:tx+roi_h, tx:tx+roi_w] = fill_value
    return mask

def crop_center(img,cropx,cropy):
    """Modified for 2dim and 3dim image
    from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image"
    """
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, ... ]
    
    

# Normalize helpers
def normalize_to_negposone(a: np.ndarray):
    # Linearly map the data values in `a`` from [a.min, a.max] to [-1,1]
    return 2.*(a - np.min(a))/np.ptp(a) -1


def normalize_negposones_to_01(a: np.ndarray):
    # Shift and rescale an array from range [-1,1] to [0,1]
    return 0.5 * (a + 1.0)

def normalize_to_01(data: np.ndarray):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

