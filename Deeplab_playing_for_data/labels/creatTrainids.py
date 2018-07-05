from __future__ import absolute_import, division, print_function
import numpy as np
import os.path
import glob
import PIL.Image as img
import matplotlib.pyplot as plt

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}


_CITYSCAPES_EVAL_ID_TO_TRAIN_ID = [255 , 255 , 255 , 255 , 255 , 255 , 255 , 0 , 1 , 255 ,
                                   255 , 2 , 3 , 4 , 255 ,255 , 255 , 5 , 255 , 6 , 7 , 8 ,
                                   9 , 10 , 11 , 12 , 13 ,14 , 15 , 255 , 255 , 16 , 17 , 18]

_IMAGE_ROOT = '/mnt/ngv/datasets/playing-for-data-cityscapes'
#_SAVE_DIR = '/home/jiarui/git/self_supervised_segmentation/Deeplab_playing_for_data/labels'

def _convert_eval_id_to_train_id(prediction, eval_id_to_train_id):
    converted_prediction = prediction.copy()
    for eval_id,train_id in enumerate(eval_id_to_train_id):
        converted_prediction[prediction == eval_id] = train_id

    return converted_prediction

def _get_files():
    pattern = '*_labelIds.png'
    searchFiles = os.path.join(_IMAGE_ROOT, 'gtFine', 'train/*', pattern)
    fileNames = glob.glob(searchFiles)
    return sorted(fileNames)

def main():
    gtFiles = _get_files()
    numImages = len(gtFiles)

    #numImages = 1

    for i in range(numImages):
        (image_name,_)=os.path.splitext(os.path.basename(gtFiles[i]))
        print(">>processing image %s of %d" % (image_name, numImages))
        #print(os.path.dirname(gtFiles[i]))

        gt = img.open(gtFiles[i])
        gt_arr = np.asarray(gt)

        gt_arr = _convert_eval_id_to_train_id(gt_arr, _CITYSCAPES_EVAL_ID_TO_TRAIN_ID )

        save_dir = os.path.dirname(gtFiles[i])
        save_name = '%s/%05d_gtFine_labelTrainIds.png' % (save_dir, i+1)
        gt = img.fromarray(gt_arr)
        gt.save(save_name)

if  __name__ == '__main__':
    main()
