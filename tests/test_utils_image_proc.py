from reprlearn.utils.image_proc import get_crop_tl
import numpy as np


def test_get_crop_tl():
    h,w = 11, 11
    img = np.ones((h,w))

    target_h, target_w = 6,6
    crop_tl = get_crop_tl(img,target_h, target_w )
    print(crop_tl)



if __name__ == '__main__':
    test_get_crop_tl()