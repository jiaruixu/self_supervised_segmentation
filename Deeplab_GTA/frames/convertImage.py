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
    'image': '_image.png',
    'prediction': '_prediction.png',
}

def _get_files(data, type):
    pattern = '*%s' % (_PATTERN_MAP[type])
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def main(_):
    # get image files
    left_image_files = _get_files('left', 'image')
    right_image_files = _get_files('right', 'image')
    left_pre_files = _get_files('left', 'prediction')
    right_pre_files = _get_files('right', 'prediction')

    num_images = len(left_image_files)
    num_images = 1

    save_dir = './frames'

    for i in range(392, num_images):
        (image_name,_) = os.path.splitext(os.path.basename(left_image_files[i]))
        print(">>processing image %s of %d" % (image_name, num_images))

        # read images
        left_image = img.open(left_image_files[i])
        right_image = img.open(right_image_files[i])
        left_pre = img.open(left_pre_files[i])
        right_pre = img.open(right_pre_files[i])

        plt.figure(figsize=(15,10))
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.97, wspace=0.03, hspace=0.07)
        plt.subplot(221)
        plt.imshow(left_image)
        plt.axis('off')
        plt.title('left image')
        plt.subplot(222)
        plt.imshow(right_image)
        plt.axis('off')
        plt.title('right image')
        plt.subplot(223)
        plt.imshow(left_pre)
        plt.axis('off')
        plt.title('left prediction')
        plt.subplot(224)
        plt.imshow(right_pre)
        plt.axis('off')
        plt.title('right prediction')
        #plt.show()
        filename = '%s/image%03d.png' % (save_dir, i)
        plt.savefig(filename)
        plt.close()

if __name__ == '__main__':
    tf.app.run()
