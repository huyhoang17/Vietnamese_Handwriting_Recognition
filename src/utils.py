import itertools

import numpy as np
from keras import backend as K

import src.config as cf


def labels_to_text(letters, labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))  # noqa


def text_to_labels(letters, text):
    return list(map(lambda x: letters.index(x), text))


def decode_batch_y_func(y_func, word_batch):
    out = y_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(cf.CHARS_, out_best)
        ret.append(outstr)
    return ret


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(cf.CHARS_, out_best)
        ret.append(outstr)
    return ret


def decode_predict_ctc(out, chars, top_paths=1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = K.get_value(
            K.ctc_decode(
                out, input_length=np.ones(out.shape[0]) * out.shape[1],
                greedy=False, beam_width=beam_width, top_paths=top_paths
            )[0][i]
        )[0]
        text = labels_to_text(chars, lables)
        results.append(text)
    return results


def predit_a_image(model_p, pimg, top_paths=1):
    # c = np.expand_dims(a.T, axis=0)
    net_out_value = model_p.predict(pimg)
    top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
    return top_pred_texts


def is_valid_str(letters, s):
    for ch in s:
        if ch not in letters:
            return False
    return True
