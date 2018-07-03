from __future__ import absolute_import, division, print_function
import numpy as np
import os.path
import glob
import itertools
import PIL.Image as img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

_FOLDER_FILES = {
    'left_recons': '/home/jiarui/git/Deeplab_KITTI/confusion matrix/left_recons',
    'ground_truth': '/home/jiarui/git/Dataset/KITTI/Segmentation/training/semantic',
}

_PATTERN_MAP = {
    'left_recons': '_10.png',
    'ground_truth': '_10.png',
}

_CITYSCAPES_EVAL_ID_TO_TRAIN_ID = [19 , 19 , 19 , 19 , 19 , 19 , 19 , 0 , 1 , 19 ,
                                   19 , 2 , 3 , 4 , 19 ,19 , 19 , 5 , 19 , 6 , 7 , 8 ,
                                   9 , 10 , 11 , 12 , 13 ,14 , 15 , 19 , 19 , 16 , 17 , 18]

LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled'
])

def _convert_eval_id_to_train_id(prediction, eval_id_to_train_id):
    converted_prediction = prediction.copy()
    for eval_id,train_id in enumerate(eval_id_to_train_id):
        converted_prediction[prediction == eval_id] = train_id
    converted_prediction[prediction == 255] = 19
    return converted_prediction

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    searchFiles = os.path.join(_FOLDER_FILES[data], pattern)
    fileNames = glob.glob(searchFiles)
    return sorted(fileNames)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)]=0.0

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=18)
    plt.colorbar()
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=14, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    #plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

def main():
    preFiles = _get_files('left_recons')
    gtFiles = _get_files('ground_truth')
    numImages = len(preFiles)

    #numImages = 1

    for i in range(numImages):
        (image_name,_)=os.path.splitext(os.path.basename(preFiles[i]))
        print(">>processing image %s " % (image_name))

        prediction = img.open(preFiles[i])
        pre_arr = np.asarray(prediction)
        gt = img.open(gtFiles[i])
        gt_arr = np.asarray(gt)
        pre_arr = _convert_eval_id_to_train_id(pre_arr, _CITYSCAPES_EVAL_ID_TO_TRAIN_ID )
        gt_arr = _convert_eval_id_to_train_id(gt_arr, _CITYSCAPES_EVAL_ID_TO_TRAIN_ID )
        gt_list = gt_arr.reshape([-1]).tolist()
        pre_list = pre_arr.reshape([-1]).tolist()
        conf_arr = confusion_matrix(gt_list, pre_list)

        labels = np.unique(np.append(gt_arr, pre_arr))
        labels_name = []
        for i in labels:
            labels_name.append(LABEL_NAMES[i])
        num_labels = len(labels)

        # Confusion matrix, without normalization
        plt.figure(figsize = (num_labels+5, num_labels))
        plot_confusion_matrix(conf_arr, classes=labels_name,
                      title='Confusion matrix, without normalization')

        save_dir = '/home/jiarui/git/Deeplab_KITTI/confusion matrix/confusion_matrix_reconstruct'
        filename = '%s/%s_wo_norm.png' % (save_dir, image_name)
        plt.savefig(filename)

        # Normalized confusion matrix
        plt.figure(figsize = (num_labels+5, num_labels))
        plot_confusion_matrix(conf_arr, classes=labels_name, normalize=True,
                      title='Normalized confusion matrix')

        filename2 = '%s/%s_norm.png' % (save_dir, image_name)
        plt.savefig(filename2)

        #plt.figure(figsize = (num_labels, num_labels))
        #plt.xlim(0,num_labels+1)
        #plt.ylim(0,num_labels+1)
        #plt.xticks(np.arange(num_labels)+1, labels_name, fontsize=14, rotation=20)
        #plt.yticks(np.arange(num_labels)+1, labels_name, fontsize=14)
        #plt.xlabel('ground truth class', fontsize=14)
        #plt.ylabel('predicted class', fontsize=14)
        #for i in np.arange(num_labels):
        #    for j in np.arange(num_labels):
        #        plt.text(i+1, j+1, conf_arr[i][j], color='k', fontsize=12)
        #plt.show()

        #image_name = os.path.basename(preFiles[i])
        #save_dir = '/home/jiarui/git/Deeplab_KITTI/confusion matrix/confusion_matrix'
        #filename = '%s/%s' % (save_dir, image_name)
        #plt.savefig(filename)


if  __name__ == '__main__':
    main()
