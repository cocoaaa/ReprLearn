# Dictionary
import argparse
from pprint import pprint
import toolz

# Create a new dictionary, sorted by the keys alphabetically
# same key:val pairs, only the order in the dictionary is changed so that key:val with 
# alphabetically-first key comes first
my_dict = {
    'b':10,
    'k':100,
    'c':20,
    'a':70
}
print('original: ', my_dict)

sorted_dict = dict(sorted(my_dict.items()))
print('sorted: ', sorted_dict)

# Use toolz lib. for conveninent and optimized 'map'ping on dictionaries
def add_10(x): return x+10
print('original: ', my_dict)
print('after applying add_10 func: ')
pprint(toolz.valmap(add_10, my_dict))

# Argparser
def argparser_workflow():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--exp_id", required=True, help="Unique id for this run."
                            " Used as outdir_root if outdir_root is None")
        # input dir or root of input dirs
        parser.add_argument("--img_dir_root", required=False, default=None, type=str, 
                            help="Data root dir containing subdirs each of which "
                            "containing images from a GM")
        parser.add_argument("--img_dir",required=False, default=None, type=str,
                            help="Path to a folder containing images from a GM")
        
        # output dir or root of output dirs
        parser.add_argument("--out_dir_root", required=False, default=None, type=str,
                            help="Root of the subdirs for fft's of images in input subdir"
                            "- Meaningful only when imgdir_root is specified.") 
        parser.add_argument("--out_dir", required=False, default=None, type=str)
        # parser.add_argument("--out_fn", required=False, default=None, type=str)
        
        # fft parameters
        parser.add_argument("--norm", required=True, type=str,
                            choices=FFT_NORM_TYPES,
                            help='Normalization mode to be used in np.fft.fft2')
        # -- to apply highpass (in pixel) before fft
        parser.add_argument("--filter_type", required=True, type=str,
                            choices=FILTER_TYPES, 
                            help="Type of filter to apply to grayscale image before FFT. "
                            "All pass applies no filtering. "
                            "For lowpass and highpass, we use median filter as low-pass "
                            "Kernel size is specified via '--kernel_size' argument.")
        parser.add_argument("-ks", "--kernel_size", default=3, type=int,
                            help='filter size for median filter for computing high-pass')
        # -- to apply pixel-value clipping (after hp) before fft
        # parser.add_argument("--plot_logscale", required=False, store_true, type=bool,
        #                     help="If set, when visualizing the avg. spectrum show it in logscale")
        # parameter for computing the average of fft's 
        parser.add_argument("-n", "--n_samples", required=False, type=int,  default=None, 
                            help="Number of images to use, per imgdir, to compute"
                            " the average of spectra. If None (default), use all images in an imgdir")
        

        # Parse cli arguments
        args = parser.parse_args()
        exp_id = args.exp_id
        filter_type = args.filter_type.lower()
        
        # Define filter functions
        allpass_func = lambda arr: arr
        highpass_func = lambda arr: arr - grayscale_median_filter(arr, kernel_size=args.kernel_size)
        lowpass_func = lambda arr: grayscale_median_filter(arr, kernel_size=args.kernel_size)

        # choose filter to apply
        if filter_type == 'allpass':
            chosen_filter = allpass_func
        elif filter_type == 'highpass':
            chosen_filter = highpass_func
        elif filter_type == 'lowpass':
            chosen_filter = lowpass_func
            
        # Define func to apply to each img fp
        norm = args.norm.lower()
        fft_function_on_fp = partial(
            compute_magnitude_spectrum_of_grayscale_fp, 
            transform=chosen_filter,
            norm=norm,
        )
        if args.img_dir_root is not None:
            img_dir_root = Path(args.img_dir_root)
            
            out_dir_root = args.out_dir_root or img_dir_root.parents / 'Output-FFT' / args.exp_id
            out_dir_root = Path(out_dir_root)
            mkdir(out_dir_root)
            
            
            compute_and_save_fft_all_subdirs(
                img_dir_root=Path(args.img_dir_root),
                out_dir_root=Path(args.out_dir_root),
                fft_function_on_fp=fft_function_on_fp,
                max_samples_per_subdir=args.n_samples
                
            )
        
        elif args.imgdir_root is None and args.img_dirpath is not None:
            img_dir = Path(args.img_dirpath)
            out_dir = args.out_dir or img_dir.parents / 'Output-FFT' / args.exp_id
            out_dir = Path(out_dir)
            mkdir(out_dir)
            
            compute_and_save_fft(
                img_dir=img_dir,
                out_dir=out_dir,
                fft_function_on_fp=fft_function_on_fp,
                max_samples=args.n_samples
            )