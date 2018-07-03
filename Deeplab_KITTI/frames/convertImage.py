import PIL.Image as img
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os.path
import glob
from xlwt import *

_FOLDERS_MAP = {
    'left': '/home/jiarui/git/Deeplab_KITTI/visulaization/leftvis',
    'est_pre': '/home/jiarui/git/Deeplab_KITTI/inconsistency/inconsistency_images',
    'pre_gt': '/home/jiarui/git/Deeplab_KITTI/inconsistency/inconsistency_images',
    'dif': '/home/jiarui/git/Deeplab_KITTI/inconsistency/inconsistency_images',
}

_PATTERN_MAP = {
    'left': '_10_image.png',
    'est_pre': '_est_pre.png',
    'pre_gt': '_pre_gt.png',
    'dif': '_dif.png',
}

LABEL_NAMES = np.asarray([
    'Red: false positive', 'Blue: false negative', 'Black: true negative', 'White: true positive'
])

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def main(_):
    # get image files
    left_image_files = _get_files('left')
    est_pre_files = _get_files('est_pre')
    pre_gt_files = _get_files('pre_gt')
    dif_files = _get_files('dif')

    num_images = len(left_image_files)
    #num_images = 1

    save_dir = '/home/jiarui/git/Deeplab_KITTI/frames/frames'

    for i in range(num_images):
        (image_name,_) = os.path.splitext(os.path.basename(left_image_files[i]))
        print(">>processing image %s " % (image_name))

        # read images
        left = img.open(left_image_files[i])
        est_pre = img.open(est_pre_files[i])
        pre_gt = img.open(pre_gt_files[i])
        dif = img.open(dif_files[i])

        plt.figure(figsize=(18,7))
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.97, wspace=0.07, hspace=0.00)
        plt.subplot(221)
        plt.imshow(left)
        plt.axis('off')
        plt.title('left image')
        plt.subplot(222)
        plt.imshow(pre_gt)
        plt.axis('off')
        plt.title('Difference between left prediction and ground truth')
        plt.subplot(223)
        plt.imshow(est_pre)
        plt.axis('off')
        plt.title('Difference between left prediction and reconstructed segmentation')
        plt.subplot(224)
        plt.imshow(dif)
        plt.axis('off')
        plt.title('Inconsistency of these two differences')
        #plt.show()
        filename = '%s/image%03d.png' % (save_dir, i)
        plt.savefig(filename)

if __name__ == '__main__':
    tf.app.run()
