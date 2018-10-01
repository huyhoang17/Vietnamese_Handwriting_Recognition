[![Build Status](https://travis-ci.org/huyhoang17/Vietnamese_Handwriting_Recognition.svg?branch=master)](https://travis-ci.org/huyhoang17/Vietnamese_Handwriting_Recognition)

# Vietnamese Handwriting Recognition


Dataset
---

Command
---

```
export PYTHONPATH=path-to-project

cp src/example_config.py src/config.py
```

Result
---

TODO
---

- Add Dockerfile
- Deploy simple demo with Tensorflow Serving
- Simple API
- Refactor code
- Add Visual Callback
- Train code with Colab's TPU
- Convert to Pytorch code

Reference
---

- Colab Notebook: https://github.com/huyhoang17/Colab_Temporary/blob/master/Training_CTC_Vietnamese_Recognition_10epochs.ipynb

CTC loss:
- https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/
- https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8
- https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

Kaggle ctc loss:
- https://dinantdatascientist.blogspot.com/2018/02/kaggle-tensorflow-speech-recognition.html

Explained ctc loss:
- https://gab41.lab41.org/speech-recognition-you-down-with-ctc-8d3b558943f0
- https://distill.pub/2017/ctc/
- https://stats.stackexchange.com/questions/320868/what-is-connectionist-temporal-classification-ctc

CTC loss param: 
- https://kur.deepgram.com/specification.html#using-ctc-loss