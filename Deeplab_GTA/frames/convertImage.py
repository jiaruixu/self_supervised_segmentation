import PIL.Image as img
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os.path
import glob
from xlwt import *

_FOLDERS_MAP = {
    'left': '../leftvis_fullseg/segmentation_results',
    'right': '../rightvis_fullseg/segmentation_results',
}

_PATTERN_MAP = {
    'left': '_prediction.png',
    'right': '_prediction.png',
}

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def main(_):
    # get image files
    left_image_files = _get_files('left')
    right_image_files = _get_files('right')

    num_images = len(left_image_files)
    #num_images = 1

    save_dir = './frames'

    for i in range(num_images):
        (image_name,_) = os.path.splitext(os.path.basename(left_image_files[i]))
        print(">>processing image %s of %d" % (image_name, num_images))

        # read images
        left = img.open(left_image_files[i])
        right = img.open(right_image_files[i])

        plt.figure(figsize=(18,7))
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.97, wspace=0.07, hspace=0.00)
        plt.subplot(121)
        plt.imshow(left)
        plt.axis('off')
        plt.title('left image')
        plt.subplot(122)
        plt.imshow(right)
        plt.axis('off')
        plt.title('right image')
        #plt.show()
        filename = '%s/image%03d.png' % (save_dir, i)
        plt.savefig(filename)

if __name__ == '__main__':
    tf.app.run()
