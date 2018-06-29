from __future__ import print_function
import glob
import os.path
import numpy as np

from PIL import Image


# A map from data type to filename postfix.
_DATA_SPLIT_MAP = {
    'train': 'training',
    'val': 'validation',
}

_DATA_FORMAT_MAP = {
    'image': 'jpg',
    'label': 'png',
}

_DATA_TYPE_MAP = {
    'image':'images',
    'label':'labels',
}

dataset_root = '/mnt/ngv/datasets/mapillary-vistas/'
train_save_dir = './training'
val_save_dir = './validation'

def _get_files(dataset_split, data):
    pattern = '*.%s' % (_DATA_FORMAT_MAP[data])
    search_files = os.path.join(dataset_root, _DATA_SPLIT_MAP[dataset_split],_DATA_TYPE_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def _resize_images(img, mask):
    assert img.size == mask.size
    w, h = img.size
    th = 1024
    tw = 2048
    if w == tw and h == th:
        return img, mask
    else:
        return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

train_image_files = _get_files('train','image')
train_label_files = _get_files('train','label')
val_image_files = _get_files('val','image')
val_label_files = _get_files('val','label')

train_num = len(train_label_files)
val_num = len(val_label_files)

#train_num = 1
#val_num = 1

print('>>processing training set')
for i in range(train_num):
    print('>>processing image %d of %d images' % (i, train_num))
    train_image = Image.open(train_image_files[i])
    train_label = Image.open(train_label_files[i])
    (image_name,_) = os.path.splitext(os.path.basename(train_image_files[i]))
    (label_name,_) = os.path.splitext(os.path.basename(train_label_files[i]))

    if image_name != label_name:
        raise RuntimeError('Name mismatched between image and label.')

    train_image, train_label = _resize_images(train_image, train_label)

    image_file_name = '%s/%s/%s.jpg' % (train_save_dir, 'images', image_name)
    train_image.save(image_file_name)

    label_array = np.array(train_label)
    label_id = Image.fromarray(label_array)

    label_file_name = '%s/%s/%s.png' % (train_save_dir, 'labels', label_name)
    label_id.save(label_file_name)

print('>>processing validation set')
for i in range(val_num):
    print('>>processing image %d of %d images' % (i, val_num))
    val_image = Image.open(val_image_files[i])
    val_label = Image.open(val_label_files[i])
    (image_name,_) = os.path.splitext(os.path.basename(val_image_files[i]))
    (label_name,_) = os.path.splitext(os.path.basename(val_label_files[i]))

    if image_name != label_name:
        raise RuntimeError('Name mismatched between image and label.')

    val_image, val_label = _resize_images(val_image, val_label)

    image_file_name = '%s/%s/%s.jpg' % (val_save_dir, 'images', image_name)
    val_image.save(image_file_name)

    label_array = np.array(val_label)
    label_id = Image.fromarray(label_array)

    label_file_name = '%s/%s/%s.png' % (val_save_dir, 'labels', label_name)
    label_id.save(label_file_name)
