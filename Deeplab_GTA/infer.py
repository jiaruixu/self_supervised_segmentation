import sys
sys.path.append('/home/jiarui/models/research/deeplab/utils')
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import get_dataset_colormap
LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP, get_dataset_colormap.get_cityscapes_name())

def main(_):
    plt.figure()
    ax = plt.gca()
    plt.imshow(FULL_COLOR_MAP.astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    plt.xticks([], [])
    ax.tick_params(width=0)
    plt.show()

if __name__=='__main__':
    tf.app.run()
