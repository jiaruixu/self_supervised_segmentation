from __future__ import absolute_import, division, print_function
import tensorflow as tf
import glob
import os.path
import sys
import PIL.Image as img
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('cityscapes_root', '/home/jiarui/Results', 'Cityscapes dataset root folder.')

_FOLDERS_MAP = {
    'right': '/home/jiarui/git/Deeplab_KITTI/confusion matrix/rightvis',
    'disparity': '/home/jiarui/git/Deeplab_KITTI/disparity/Disparity',
}

_PATTERN_MAP = {
    'right': '_10.png',
    'disparity': '_10.png',
}

def bilinear_sampler_1d_h(input_images, x_offset, depth_map, wrap_mode='edge', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _map(x_t_row_array, depth_row_array):
        ind = np.array([], dtype = 'int32')
        x_t_row_list = x_t_row_array.tolist()
        depth_row_list = depth_row_array.tolist()
        for item in list(set(x_t_row_list)):
            if x_t_row_list.count(item) > 1:
                idx = np.where(x_t_row_array==item)[0]
                minval = np.min(depth_row_array[idx])
                idx = idx[np.where(depth_row_array[idx] != minval)]
                ind = np.concatenate((ind,idx))
        return sorted(ind)

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            _edge_size = 0

            x = np.clip(x, 0.0, _width_f - 1)

            x0_f = np.floor(x)
            y0_f = np.floor(y)
            x1_f = x0_f + 1

            x0 = x0_f.astype(np.int32)
            y0 = y0_f.astype(np.int32)

            x1 = np.clip(x1_f, 0.0, _width_f - 1).astype(np.int32)

            dim2 = _width
            dim1 = _width * _height

            base = np.zeros([1, _height * _width], np.int32)
            base = np.reshape(base, [-1])
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = np.reshape(im, [-1])

            pix_l = im_flat[idx_l]
            pix_r = im_flat[idx_r]

            weight_l = x1_f - x
            weight_r = x - x0_f

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = np.meshgrid(np.arange(_width), np.arange(_height))

            x_t_flat = x_t.ravel()
            y_t_flat = y_t.ravel()

            #x_offset_unlabel_idx = tf.reshape(tf.where(tf.less_equal(tf.reshape(x_offset, [-1]), 0)), [-1])
            #x_offset_unlabel_idx = [i for i in x_offset.ravel().tolist() if i <= 0]

            x_t_flat = x_t_flat - x_offset.ravel()

            x_t_flat_clip = np.clip(x_t_flat, 0.0, _width_f - 1)
            x0_stack = np.reshape(x_t_flat_clip, [_height, _width])
            depth_array = np.reshape(_depth_map, [_height, _width])

            x0_array = np.floor(x0_stack)

            ind = np.array([], dtype = 'int32')
            for i in np.arange(_height):
                val = _map(x0_array[i], depth_array[i])
                ind = np.concatenate((ind, val + i*_width))


            #input_transformed, ind = _interpolate(input_images, x_t_flat, y_t_flat)
            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            #print(ind)

            input_transformed[ind.astype(np.int32)] = 255
            output = np.reshape(input_transformed, [_height, _width])

            return output

    with tf.variable_scope(name):
        # _num_batch    = tf.shape(input_images)[0]
        _height       = np.shape(input_images)[0]
        _width        = np.shape(input_images)[1]
        #_num_channels = tf.shape(input_images)[2]

        _height_f = float(_height)
        _width_f  = float(_width)

        _wrap_mode = wrap_mode

        _depth_map = depth_map

        output = _transform(input_images, x_offset)
        return output

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def main(_):
    right_image_files = _get_files('right')
    disparity_map_files = _get_files('disparity')
    num_images = len(right_image_files)

    #num_images = 1

    for i in range(num_images):
        (image_name,_)=os.path.splitext(os.path.basename(right_image_files[i]))
        print(">>processing image %s " % (image_name))

        right_image  = img.open(right_image_files[i])
        right_arr  = np.asarray(right_image ).astype(np.float32)
        #print(right_arr)

        disparity_image  = img.open(disparity_map_files[i])
        disparity_map  = np.asarray(disparity_image).astype(np.float32)*228.0 / 255.0
        # Compute depth. The depth calculated is propotional to real depth
        depth_map = 1.0 / disparity_map

        left_est = bilinear_sampler_1d_h(right_arr, disparity_map, depth_map)
        #print(left_est)
        #left_est_summary = tf.expand_dims(left_est, 0)

        save_dir = '/home/jiarui/git/Deeplab_KITTI/confusion matrix/left_recons'
        filename = '%s/%s.png' % (save_dir, image_name)
        left_image = img.fromarray(left_est.astype(np.uint8))
        left_image.save(filename)

if __name__ == '__main__':
    tf.app.run()
