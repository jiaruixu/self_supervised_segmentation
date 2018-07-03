import PIL.Image as img
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os.path
import glob
from xlwt import *
from xlrd import *

def main(_):
    wb = open_workbook('/home/jiarui/git/Deeplab_KITTI/inconsistency/data.xlsx')
    sh = wb.sheet_by_index(0)
    name = sh.col_values(0,1)
    re_pre = sh.col_values(5,1)
    pre_gt = sh.col_values(6,1)
    num = len(name)
    bar_width = 0.35
    index = np.arange(num)
    ind_sort = np.argsort(re_pre)
    ind_show = ind_sort[::10]
    index_show = index[::10]
    re_pre_sort = []
    for i in np.argsort(pre_gt):
        re_pre_sort.append(re_pre[i])

    plt.figure(figsize=(18,7))
    plt.bar(index, re_pre, bar_width, color = 'r', label = 're_pre_error')
    plt.bar(index+bar_width, sorted(pre_gt), bar_width, color = 'b', label = 'pre_gt_error')
    #plt.bar(index, sorted(re_pre), bar_width)
    plt.xticks(index_show, ind_show)
    plt.legend()
    plt.title('Difference between reconstructed image and left prediction vs Difference between left prediction and ground truth')
    plt.xlabel('Image number')
    plt.ylabel('Error rate')
    #plt.show()
    fileName = '/home/jiarui/git/Deeplab_KITTI/inconsistency/errorCorrelation.png'
    plt.savefig(fileName)


if __name__=='__main__':
    tf.app.run()
