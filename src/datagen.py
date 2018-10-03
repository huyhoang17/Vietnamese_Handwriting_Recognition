import os
import pickle

import cv2
import scipy.misc

import src.config as cf
from src.data_utils import gen_data


def gen_data_augment():
    with open(cf.LABELS, 'rb') as f:  # noqa
        samples = pickle.load(f)

    fn_paths, gt_texts = [], []
    maps = {}
    for sample in samples:
        img = list(sample.keys())[0]
        fn_path = os.path.join(cf.RAW_DATA, img)
        fn_paths.append(fn_path)
        gt_texts.append(list(sample.values())[0])

    for ind, fn_path in enumerate(fn_paths):
        fn = fn_path.split('/')[-1]
        maps[fn] = gt_texts[ind]
        scipy.misc.imsave(
            os.path.join(
                cf.GEN_DATA, fn)
        )
        main_img = cv2.imread(fn_path, cv2.IMREAD_GRAYSCALE)
        fns = gen_data(
            cf.GEN_DATA, main_img, fn, reversed_img=True,
            is_save=True, return_img=False
        )
        for fn_ in fns:
            maps[fn_] = gt_texts[ind]

    pickle.dump(maps, '../data/gen_labels.pkl')


if __name__ == '__main__':
    gen_data_augment()
