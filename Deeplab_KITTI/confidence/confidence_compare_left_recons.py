import PIL.Image as img
import numpy as np
import os.path
import glob
import os.path

_FOLDER_FILES = {
    'left_prediction': './leftvis',
    'recon_left': './reconstruct_left_confidence',
    'dif': '../inconsistency/inconsistency_images'
}

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    searchFiles = os.path.join(_FOLDER_FILES[data], pattern)
    fileNames = glob.glob(searchFiles)
    return sorted(fileNames)

def main(_):
    fileName_left = _get_files('left_prediction')
    fileName_recon_left = _get_files('recon_left')
    dif_files = _get_files('dif')
    num_files = len(fileName_left)

    num_files = 1

    for i in range(num_files):
        (image_name,_) = os.path.splitext(os.path.basename(fileName_left[i]))
        print('>>processing image %s' % (image_name))
        conf_left = np.loadtxt(fileName_left[i])
        conf_releft = np.loadtxt(fileName_recon_left[i])
        dif = img.open(dif_files[i])

        dif_array = np.asarray(dif).reshape([-1, 3])
        dif_list = dif_array.tolist()
        ind = [i for i in range(len(dif_list)) if dif_list[i] == [255, 0, 0]]

        left_flat = conf_left.reshape([-1])
        releft_flat = conf_releft.reshape([-1])

        [h, w, ch] = np.shape(dif)
        dif_green = list(dif_list)

        for j in ind:
            if np.abs(left_flat[j] - releft_flat[j]) < 5:
                dif_green[j] = [0, 255, 0]


        dif_new = np.asarray(dif_green, dtype = np.uint8).reshape([h, w, ch])
        dif_show = img.fromarray(dif_new)

        save_image_name = './confidence_compare/%s_compare.png' % (image_name)
        dif_show.save(save_image_name)


if __name__=='__main__':
    main()
