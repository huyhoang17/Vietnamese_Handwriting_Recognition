import editdistance
import numpy as np
import keras

import src.config as cf
from src.utils import decode_batch, labels_to_text
from src.log import get_logger

logger = get_logger(__name__)


class VizCallback(keras.callbacks.Callback):

    def __init__(self, y_func, test_set,
                 text_size=256, num_display_words=6,
                 set_index_batch=False):
        """
        :param y_func:
        :param test_set: test_set generator
        :param text_size: no test set samples, default is 256
        :param num_display_words: no words to display
        """
        self.y_func = y_func
        self.test_set = test_set
        self.text_size = text_size
        self.num_display_words = num_display_words
        self.set_index_batch = set_index_batch

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        index_batch = 0

        while num_left > 0:
            # no samples per batch
            word_batch = self.test_set[index_batch][0]

            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(
                    decoded_res[j], word_batch['source_str'][j]
                )
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / \
                    len(word_batch['source_str'][j])

            num_left -= num_proc
            index_batch += 1
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        logger.info(
            '\nOut of %d samples:  Mean edit distance: %.3f'
            'Mean normalized edit distance: %0.3f',
            num, mean_ed, mean_norm_ed
        )

    def on_epoch_end(self, epoch, logs={}):
        # get inputs dict
        if self.set_index_batch:
            max_index_batch = len(self.test_set.ids) // cf.BATCH_SIZE
            index_batch = np.random.randint(0, max_index_batch)
        else:
            # DEFAULT: get first batch
            index_batch = 0
        batch = self.test_set[index_batch][0]

        inputs = batch['the_inputs'][:self.num_display_words]
        labels = batch['the_labels'][:self.num_display_words].astype(np.int32)
        labels = [labels_to_text(cf.CHARS_, label) for label in labels]

        # feature vectors after softmax
        pred = self.y_func([inputs])
        logger.info("Pred shape: ", pred.shape)
        pred = pred[0]
        pred_texts = decode_batch(cf.CHARS_, pred)
        for i in range(min(self.num_display_words, len(inputs))):
            logger.info("label: %s - predict: %s", labels[i], pred_texts[i])

        self.show_edit_distance(self.text_size)
