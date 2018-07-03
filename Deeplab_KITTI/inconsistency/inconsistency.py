import PIL.Image as img
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os.path
import glob
from xlwt import *

_FOLDERS_MAP = {
    'left': '/home/jiarui/git/Deeplab_KITTI/visulaization/leftvis',
    'left_est': '/home/jiarui/git/Deeplab_KITTI/reconstruct/reconstruct_left',
    'gt': '/home/jiarui/git/Dataset/KITTI/Segmentation/training/semantic_rgb',
}

_PATTERN_MAP = {
    'left': '_10_prediction.png',
    'left_est': '_10_prediction.png',
    'gt': '_10.png',
}

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def _reconstruct_prediction_difference(left_est_array, left_image_array, gt_mask_idx, left_est_mask_idx):
    # inconsistency between reconstructed left image based on mccnn disparity and left prediction image
    est_pre_dif = np.abs(left_est_array - left_image_array)
    [h, w, ch] = np.shape(est_pre_dif)
    est_pre_list = est_pre_dif.reshape([-1, 3]).tolist()
    for i in range(len(est_pre_list)):
        if est_pre_list[i] != [0, 0, 0]:
            est_pre_list[i] = [255, 255, 255]

    for i in gt_mask_idx:
        est_pre_list[i] = [0, 0, 0]

    for i in left_est_mask_idx:
        est_pre_list[i] = [0, 0, 0]

    error = est_pre_list.count([255,255, 255])
    est_pre_error = float(error) / (h*w)

    est_pre_dif = np.asarray(est_pre_list, dtype = np.uint8).reshape([h, w, ch])
    return est_pre_dif, est_pre_error

def _prediction_gt_difference(left_image_array, gt_array, gt_mask_idx, left_est_mask_idx):
    # inconsistency between left prediction image and ground truth
    pre_gt_dif = np.abs(left_image_array - gt_array)

    [h, w, ch] = np.shape(pre_gt_dif)
    pre_gt_list = pre_gt_dif.reshape([-1, 3]).tolist()
    for i in range(len(pre_gt_list)):
        if pre_gt_list[i] != [0, 0, 0]:
            pre_gt_list[i] = [255, 255, 255]

    for i in gt_mask_idx:
        pre_gt_list[i] = [0, 0, 0]

    error= pre_gt_list.count([255,255, 255])
    pre_gt_error = float(error) / (h*w)
    #print(pre_gt_acc)

    pre_gt_dif = np.asarray(pre_gt_list, dtype = np.uint8).reshape([h, w, ch])
    return pre_gt_dif, pre_gt_error

def _reconstruct_gt_difference(left_est_array, gt_array, gt_mask_idx, left_est_mask_idx):
    # inconsistency between reconstructed left image based on mccnn disparity and ground truth
    est_gt_dif = np.abs(left_est_array - gt_array)

    [h, w, ch] = np.shape(est_gt_dif)
    est_gt_list = est_gt_dif.reshape([-1, 3]).tolist()
    for i in range(len(est_gt_list)):
        if est_gt_list[i] != [0, 0, 0]:
            est_gt_list[i] = [255, 255, 255]

    for i in gt_mask_idx:
        est_gt_list[i] = [0, 0, 0]

    for i in left_est_mask_idx:
        est_gt_list[i] = [0, 0, 0]

    error = est_gt_list.count([255, 255, 255])
    est_gt_error = float(error) / (h*w)
    #print('left image based on mccnn disparity and ground truth:', est_gt_acc)

    est_gt_dif = np.asarray(est_gt_list, dtype = np.uint8).reshape([h, w, ch])
    return est_gt_dif, est_gt_error

def _inconsistency_difference(est_pre_dif, pre_gt_dif, image_name):
    [h, w, ch] = np.shape(est_pre_dif)
    est_pre_list = est_pre_dif.reshape([-1, 3]).tolist()
    pre_gt_list = pre_gt_dif.reshape([-1, 3]).tolist()
    dif_list = list(est_pre_list)
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in range(len(est_pre_list)):
        if est_pre_list[i] == [0, 0, 0] and pre_gt_list[i] == [0, 0, 0]:
            dif_list[i] = [0, 0, 0] # true negative
            tn = tn + 1
        elif est_pre_list[i] == [255, 255, 255] and pre_gt_list[i] == [0, 0, 0]:
            dif_list[i] = [255, 0, 0] # false positive, red
            fp = fp + 1
        elif est_pre_list[i] == [0, 0, 0] and pre_gt_list[i] == [255, 255, 255]:
            dif_list[i] = [0, 0, 255] # false negative, blue
            fn = fn + 1
        elif est_pre_list[i] == [255, 255, 255] and pre_gt_list[i] == [255, 255, 255]:
            dif_list[i] = [255, 255, 255] # true positive
            tp = tp + 1

    tn_per = float(tn)/(tn + fp + fn + tp)
    fp_per = float(fp)/(tn + fp + fn + tp)
    fn_per = float(fn)/(tn + fp + fn + tp)
    tp_per = float(tp)/(tn + fp + fn + tp)
    error_rate = [image_name, tn_per, fp_per, fn_per, tp_per]

    dif = np.asarray(dif_list, dtype = np.uint8).reshape([h, w, ch])
    return dif, error_rate


def main(_):
    # get image files
    left_est_files = _get_files('left_est') # reconstructed left prediction
    left_image_files = _get_files('left') # left prediction
    gt_files = _get_files('gt') # ground truth
    num_images = len(left_image_files)

    # initialization
    title = ["Image Name", "True Negative", "False Positive", "False Negative", "True Positive", "reonstruction prediction error", "prediction gt error", "reconstruction gt error"]
    data_list = [title]

    for i in range(num_images):

        (image_name,_) = os.path.splitext(os.path.basename(left_image_files[i]))
        print(">>processing image %s " % (image_name))

        # read images
        left_est = img.open(left_est_files[i])
        left_image = img.open(left_image_files[i])
        gt_image = img.open(gt_files[i])

        # crop the images
        area = (50, 0, 1242, 375)
        left_est_crop = left_est.crop(area)
        left_image_crop = left_image.crop(area)
        gt_crop = gt_image.crop(area)

        # convert to array
        left_est_array = np.asarray(left_est_crop)
        left_image_array = np.asarray(left_image_crop)
        gt_array = np.asarray(gt_crop)

        # get unlabeled mask
        gt_list = gt_array.reshape([-1, 3]).tolist()
        gt_mask_idx = [item for item in range(len(gt_list)) if gt_list[item] == [0, 0, 0]]
        left_est_list = left_est_array.reshape([-1, 3]).tolist()
        left_est_mask_idx = [item for item in range(len(left_est_list)) if left_est_list[item] == [0, 0, 0]]

        # inconsistency between reconstructed left image and left prediction image
        est_pre_dif, est_pre_error = _reconstruct_prediction_difference(left_est_array, left_image_array, gt_mask_idx, left_est_mask_idx)

        # inconsistency between left prediction image and ground truth
        pre_gt_dif, pre_gt_error = _prediction_gt_difference(left_image_array, gt_array, gt_mask_idx, left_est_mask_idx)

        # inconsistency between reconstructed left image based and ground truth
        est_gt_dif, est_gt_error = _reconstruct_gt_difference(left_est_array, gt_array, gt_mask_idx, left_est_mask_idx)

        # est_pre and pre_gt, false positive, false negative, true positive, true negative
        dif, row = _inconsistency_difference(est_pre_dif, pre_gt_dif, image_name)

        row.append(est_pre_error)
        row.append(pre_gt_error)
        row.append(est_gt_error)
        data_list.append(row)

        # convert to images
        est_pre_show = img.fromarray(est_pre_dif)
        pre_gt_show = img.fromarray(pre_gt_dif)
        est_gt_show = img.fromarray(est_gt_dif)
        dif_show = img.fromarray(dif)

        # save inconsistency images
        save_dir = '/home/jiarui/git/Deeplab_KITTI/inconsistency/inconsistency_images'
        filename1 = '%s/%s_%s' % (save_dir, image_name, 'est_pre.png')
        est_pre_show.save(filename1)
        filename2 = '%s/%s_%s' % (save_dir, image_name, 'pre_gt.png')
        pre_gt_show.save(filename2)
        filename3 = '%s/%s_%s' % (save_dir, image_name, 'est_gt.png')
        est_gt_show.save(filename3)
        filename4 = '%s/%s_%s' % (save_dir, image_name, 'dif.png')
        dif_show.save(filename4)

    # save to excel
    file = Workbook(encoding='utf-8')
    table = file.add_sheet('data')
    for row, content in enumerate(data_list):
        for col, item in enumerate(content):
            table.write(row, col, item)
    tableName = '/home/jiarui/git/Deeplab_KITTI/inconsistency/data.xlsx'
    file.save(tableName)


if __name__=='__main__':
    tf.app.run()
