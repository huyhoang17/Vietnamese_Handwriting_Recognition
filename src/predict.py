import pickle

import numpy as np
from keras.models import Model, model_from_json

import src.config as cf
from src.loaders import TextSequenceGenerator
from src.log import get_logger
from src.utils import decode_predict_ctc, labels_to_text

logger = get_logger(__name__)


def load_trained_model():
    with open(cf.CONFIG_MODEL) as f:
        json_string = f.read()

    model = model_from_json(json_string)
    model.load_weights(cf.WEIGHT_MODEL)\

    return model


def load_test_samples():
    with open(cf.LABELS_FILE, 'rb') as f:  # noqa
        data = pickle.load(f)
    no_samples = len(data)
    no_train_set = int(no_samples * cf.TRAIN_SIZE)
    test_set = TextSequenceGenerator(
        data[no_train_set:],
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=cf.DOWNSAMPLE_FACTOR,
        shuffle=False
    )
    del data
    return test_set


def predict(test_set, index_batch, index_img):
    model = load_trained_model()

    input_data = model.get_layer('the_input').output
    y_pred = model.get_layer('softmax').output
    model_p = Model(inputs=input_data, outputs=y_pred)

    samples = test_set[index_batch]
    img = samples[0]['the_input'][index_img]
    img = np.expand_dims(img, axis=0)
    logger.info(img.shape)

    net_out_value = model_p.predict(img)
    logger.info(net_out_value.shape)

    pred_texts = decode_predict_ctc(net_out_value, cf.CHARS)
    logger.info(pred_texts[0])
    gt_texts = test_set[index_batch][0]['the_labels'][index_img]
    gt_texts = labels_to_text(cf.CHARS, gt_texts.astype(int))
    logger.info(gt_texts)


if __name__ == '__main__':
    import random  # noqa
    rd_index_batch = random.randint(0, 10)
    test_set = load_test_samples()
    for i in range(cf.BATCH_SIZE):
        predict(test_set, rd_index_batch, i)
