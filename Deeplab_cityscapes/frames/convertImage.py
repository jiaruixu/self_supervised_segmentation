import PIL.Image as img
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os.path
import glob
from xlwt import *

_FOLDERS_MAP = {
    'image': '../visualization/leftvis',
}

_PATTERN_MAP = {
    'image': '_prediction.png',
}


def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def main(_):
    # get image files
    image_files = _get_files('image')

    #num_images = len(image_files)
    num_images = 400

    save_dir = './frames'

    for i in range(num_images):
        (image_name,_) = os.path.splitext(os.path.basename(image_files[i]))
        print(">>processing image %s " % (image_name))

        # read images
        image = img.open(image_files[i])
        #plt.show()
        filename = '%s/image%03d.png' % (save_dir, i)
        image.save(filename)

if __name__ == '__main__':
    tf.app.run()
