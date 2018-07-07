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

# Colormap (id, R, G, B) based on `https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py`
cmap = np.array([[  0, 128,  64, 128],
                 [  1, 244,  35, 232],
                 [  2,  70,  70,  70],
                 [  3, 102, 102, 156],
                 [  4, 190, 153, 153],
                 [  5, 153, 153, 153],
                 [  6, 250, 170,  30],
                 [  7, 220, 220,   0],
                 [  8, 107, 142,  35],
                 [  9, 152, 251, 152],
                 [ 10,  70, 130, 180],
                 [ 11, 220,  20,  60],
                 [ 12, 255,   0,   0],
                 [ 13,   0,   0, 142],
                 [ 14,   0,   0,  70],
                 [ 15,   0,  60, 100],
                 [ 16,   0,  80, 100],
                 [ 17,   0,   0, 230],
                 [ 18, 119,  11,  32]], dtype=np.uint8)


#bin_size = 1000

#fn_lbl = sorted(glob('{}/*.png'.format(label_dir)))
#fn_img = sorted(glob('{}/*.png'.format(image_dir)))
#if not len(fn_img) == len(fn_lbl):
#    raise ValueError('Number of images and labels must be tha same.')

def convertLabel2TrainIds(fn_lbl):
#for i, name in enumerate(zip(fn_img, fn_lbl)):
    #name_img, name_lbl = name
    name_lbl = fn_lbl
    #if not os.path.basename(name_img) == os.path.basename(name_lbl):
    #    raise Waring('[!] Names for `leftImg8bit` and `gtFine` do not match.')\

    #print('[{:6d}/{:6d}]'.format(i, len(fn_img)))
    print(name_lbl)

    lbl_rgb = imread(name_lbl)
    lbl_id = np.zeros_like(lbl_rgb[:, :, 0])
    lbl_id[:, :] = 255
    for k in range(cmap.shape[0]):
        r = (lbl_rgb[:, :, 0] == cmap[k, 1])
        g = (lbl_rgb[:, :, 1] == cmap[k, 2])
        b = (lbl_rgb[:, :, 2] == cmap[k, 3])

        #lbl_id += cmap[k, 0] * (r & g & b).astype(np.uint8)
        lbl_id[r & g & b] = cmap[k, 0]

    #folder_gtFine = os.path.join(target_dir, 'gtFine/train', '{:06d}'.format(i // bin_size))
    #folder_leftImg8bit = folder_gtFine.replace('gtFine', 'leftImg8bit')

    #for folder in [folder_gtFine, folder_leftImg8bit]:
    #    if not os.path.exists(folder):
    #        os.makedirs(folder)

    folder_gtFine = os.path.join(target_dir, 'gtFine/train')
    if not os.path.exists(folder_gtFine):
        os.makedirs(folder_gtFine)

    imsave(os.path.join(folder_gtFine, os.path.basename(name_lbl).replace('.png', '_gtFine_labelTrainIds.png')), lbl_id)

    copyfile(name_lbl, os.path.join(folder_gtFine, os.path.basename(name_lbl).replace('.png', '_color.png')))
    #copyfile(name_img, os.path.join(folder_leftImg8bit, os.path.basename(name_img).replace('.png', '_leftImg8bit.png')))

if __name__ == '__main__':
    p = Pool(8)
    fn_lbl = sorted(glob('{}/*.png'.format(label_dir)))
    p.map(convertLabel2TrainIds, fn_lbl)
