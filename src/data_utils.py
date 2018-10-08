import os

import cv2
import numpy as np
from PIL import Image
import scipy.misc

import src.config as cf
from src.log import get_logger

logger = get_logger(__name__)


def check_eq(img, img2):
    return np.array_equal(img, img2)


def crop(img):
    y_min = np.min(np.where(img > cf.PIXEL_INDEX)[0])
    y_max = np.max(np.where(img > cf.PIXEL_INDEX)[0])
    x_min = np.min(np.where(img > cf.PIXEL_INDEX)[1])
    x_max = np.max(np.where(img > cf.PIXEL_INDEX)[1])
    img = img[y_min:y_max, x_min:x_max]

    return img


"""
Reference: https://github.com/josarajar/HTRTF/blob/master/Modules/DataAugmentation.py  # noqa
"""


def scale(img, scale_prob=0.5, scale_stdv=0.01):
    scale = np.random.binomial(1, scale_prob)
    if scale:
        imgPIL = Image.fromarray(img)
        ho, vo = imgPIL.size
        scale_factor = np.random.lognormal(sigma=scale_stdv)
        hn, vn = int(scale_factor * ho), int(scale_factor * vo)
        img_sc = imgPIL.resize((hn, vn))

        img = np.array(img_sc).reshape((vn, hn))

        if hn > ho:
            img = img[int(vn / 2) - int(vo / 2):int(vn / 2) +
                      int(np.ceil(vo / 2)),
                      int(hn / 2) - int(ho / 2):int(hn / 2) +
                      int(np.ceil(ho / 2))]
        else:
            img = np.pad(
                img, ((int((vo - vn) / 2), int(np.ceil((vo - vn) / 2))),
                      ((int((ho - hn) / 2), int(np.ceil((ho - hn) / 2))))),
                mode='constant'
            )

    return img


def shear(img, shear_prob=0.5, shear_prec=150):
    shear = np.random.binomial(1, shear_prob)

    if shear:
        rows, cols = img.shape
        shear_angle = np.random.vonmises(0, kappa=shear_prec)
        m = np.tan(shear_angle)

        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[50, 50], [200, 50], [50 + m * 150, 200]])
        M = cv2.getAffineTransform(pts1, pts2)

        img = cv2.warpAffine(img, M, (cols, rows))

    return img


def rotate(img, rotate_prob=0.5, rotate_prec=250):
    rotate = np.random.binomial(1, rotate_prob)

    if rotate:
        rows, cols = img.shape
        rotate_prec = rotate_prec * max(rows / cols, cols / rows)
        rotate_angle = np.random.vonmises(0, kappa=rotate_prec) * 180 / np.pi
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

    return img


def translate(img, translate_prob=0.5, translate_stdv=0.005):
    translate = np.random.binomial(1, translate_prob)

    if translate:
        rows, cols = img.shape
        h_translation_factor = np.random.normal(0, scale=translate_stdv * cols)
        v_translation_factor = np.random.normal(0, scale=translate_stdv * rows)
        M = np.float32([[1, 0, h_translation_factor],
                        [0, 1, v_translation_factor]])
        img = cv2.warpAffine(img, M, (cols, rows))

    return img


def dilate(img, dilation_prob=0.5, dilation_srate=0.99, dilation_rrate=1):

    dilate = np.random.binomial(1, dilation_prob)

    if dilate:
        kernel_size = np.min([2 * np.random.geometric(dilation_srate) + 1, 15])
        kernel = np.zeros([kernel_size, kernel_size])
        center = np.array([int(kernel_size / 2), int(kernel_size / 2)])
        for x in range(kernel_size):
            for y in range(kernel_size):
                d = np.linalg.norm(np.array([x, y]) - center)
                p = np.exp(-d * 1)
                value = np.random.binomial(1, p)
                kernel[x, y] = value or 10**-16

        img = cv2.dilate(img, kernel, iterations=1)

    return img


def erode(img, erosion_prob=0.5, erosion_srate=1, erosion_rrate=1.2):

    erode = np.random.binomial(1, erosion_prob)

    if erode:
        kernel_size = np.min([2 * np.random.geometric(erosion_srate) + 1, 15])
        kernel = np.zeros([kernel_size, kernel_size])
        center = np.array([int(kernel_size / 2), int(kernel_size / 2)])
        for x in range(kernel_size):
            for y in range(kernel_size):
                d = np.linalg.norm(np.array([x, y]) - center)
                p = np.exp(-d * 1)
                value = np.random.binomial(1, p)
                kernel[x, y] = value or 10**-16

        img = cv2.erode(img, kernel, iterations=1)

    return img


def gen_data(path_dir, img, fn, reversed_img=True,
             is_save=False, return_img=False):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img, Image):
        img = Image.fromarray(img)

    logger.info("size: ", img.shape)
    if reversed_img:
        # convert from (bg: w, text: b) to (bg: b, text: w)
        img = 255 - img

    imgs = []
    fns = []
    name = fn.split('.')[0]
    suffix = fn.split('.')[1]
    for ind in range(1, cf.NO_GEN_IMAGES + 1):
        img_np = translate(img)
        img_np = rotate(img_np)
        img_np = shear(img_np)
        img_np = scale(img_np)
        img_np = dilate(img_np)
        # img_np = erode(img_np)
        imgs.append(img_np)

        fn_new = name + "_{}".format(ind) + '.' + suffix
        if is_save:
            scipy.misc.imsave(os.path.join(
                path_dir, fn_new), img_np
            )
        fns.append(fn_new)
    if return_img:
        return imgs, fns
    else:
        return fns
