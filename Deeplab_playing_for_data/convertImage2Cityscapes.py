import os
from glob import glob
from scipy.misc import imread, imsave
from shutil import copyfile
import numpy as np
from multiprocessing import Pool

parent_dir = '/mnt/fcav/datasets/playing-for-data'
image_dir = '{}/images'.format(parent_dir)
label_dir = '{}/labels'.format(parent_dir)
target_dir = '{}/cityscapes_format'.format(parent_dir)

#bin_size = 1000

#fn_lbl = sorted(glob('{}/*.png'.format(label_dir)))
#fn_img = sorted(glob('{}/*.png'.format(image_dir)))
#if not len(fn_img) == len(fn_lbl):
#    raise ValueError('Number of images and labels must be tha same.')

def convertImageName(name_img):
    #if not os.path.basename(name_img) == os.path.basename(name_lbl):
    #    raise Waring('[!] Names for `leftImg8bit` and `gtFine` do not match.')\

    print(name_img)

    folder_leftImg8bit = os.path.join(target_dir, 'leftImg8bit/train')
    if not os.path.exists(folder_leftImg8bit):
        os.makedirs(folder_leftImg8bit)

    copyfile(name_img, os.path.join(folder_leftImg8bit, os.path.basename(name_img).replace('.png', '_leftImg8bit.png')))

if __name__ == '__main__':
    fn_img = sorted(glob('{}/*.png'.format(image_dir)))
    #fn_lbl = sorted(glob('{}/*.png'.format(label_dir)))
    #if not len(fn_img) == len(fn_lbl):
    #    raise ValueError('Number of images and labels must be tha same.')

    p = Pool(8)
    p.map(convertImageName, fn_img[12500:])
