import os

import numpy as np
import keras
from keras import backend as K
import keras.callbacks
import cv2

import src.config as cf
from src.utils import text_to_labels
from src.log import get_logger

logger = get_logger(__name__)


class TextSequenceGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, samples, batch_size=16,
                 img_size=cf.IMAGE_SIZE, max_text_len=256,
                 downsample_factor=4, shuffle=True):
        # train 95, test 5
        imgs, gt_texts = [], []
        for sample in samples:
            img = list(sample.keys())[0]
            fn_path = os.path.join('path-to-vn-data-dir', img.split('/')[-1])
            imgs.append(fn_path)
            gt_texts.append(list(sample.values())[0])
        self.imgs = imgs
        self.gt_texts = gt_texts

        self.max_text_len = max_text_len
        self.chars = cf.CHARS_
        self.blank_label = len(self.chars)
        self.ids = range(len(self.imgs))

        self.img_size = img_size
        self.img_w, self.img_h = self.img_size
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        ids = [self.ids[k] for k in indexes]

        # some images can not read with cv2.imread
        # for id_ in [1820, 5915]:
        #     if id_ in ids:
        #         ids.remove(id_)
        X, y = self.__data_generation(ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        """Generates data containing batch_size samples"""
        for i, id_ in enumerate(ids):
            img = cv2.imread(self.imgs[id_], cv2.IMREAD_GRAYSCALE)
            if img is None:
                ids.remove(id_)
                logger.info("\n==> Error id: ", id_)
        size = len(ids)

        if K.image_data_format() == 'channels_first':
            X = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X = np.ones([size, self.img_w, self.img_h, 1])
        Y = np.ones([size, self.max_text_len])
        # input_length = np.ones((size, 1), dtype=np.float32) * \
        #     (self.img_w // self.downsample_factor - 2)
        input_length = np.ones((size, 1), dtype=np.float32) * 254
        label_length = np.zeros((size, 1), dtype=np.float32)

        # Generate data
        for i, id_ in enumerate(ids):

            img = cv2.imread(self.imgs[id_], cv2.IMREAD_GRAYSCALE)  # (h, w)
            if img is None:
                continue
            # img = 255 - img  # bg: black, text: white
            # bg: white, text: black
            ratio = img.shape[0] / self.img_h
            new_w = int(img.shape[1] / ratio) + 1
            resized_image = cv2.resize(img, (new_w, self.img_h))  # (h, w)
            img = cv2.copyMakeBorder(
                resized_image, 0, 0, 0, self.img_w - resized_image.shape[1],
                cv2.BORDER_CONSTANT, value=0
            )  # (h, w)
            img = img / 255  # (h, w)

            if K.image_data_format() == 'channels_first':
                img = np.expand_dims(img, 0)  # (1, h, w)
                img = np.expand_dims((0, 2, 1))  # (1, w, h)
            else:
                img = np.expand_dims(img, -1)  # (h, w, 1)
                img = img.transpose((1, 0, 2))  # (w, h, 1)

            X[i] = img
            text2label = text_to_labels(self.chars, self.gt_texts[id_])
            Y[i] = text2label + \
                [self.blank_label for _ in range(
                    self.max_text_len - len(text2label))]
            label_length[i] = len(self.gt_texts[id_])

        inputs = {
            'the_input': X,
            'the_labels': Y,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([size])}

        return inputs, outputs
