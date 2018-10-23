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

- Model trained with 25 epochs: https://github.com/huyhoang17/Colab_Temporary/blob/master/Training_CTC_Vietnamese_Recognition_25epochs.ipynb

- [Overfitting model] Model trained with 10 epochs (initial weights from 25 epochs), reduced LR from 0.02 to 0.005: https://github.com/huyhoang17/Colab_Temporary/blob/master/%5BOverfitting%5DTraining_CTC_Vietnamese_Recognition_40epochs.ipynb

TODO
---

- Add Dockerfile
- Deploy simple demo with Tensorflow Serving
- Simple API
- Refactor code
- Add Visual Callback :heavy_check_mark:
    - [callbacks.py](src/callbacks.py)
- Train code with Colab's TPU?
- Convert to Pytorch code

Prevent Overfitting
---

- Data Augmentation :heavy_check_mark:
    - [data_utils.py](src/data_utils.py) and [datagen.py](src/datagen.py)
- Generate more data
	- https://arxiv.org/abs/1308.0850
	- https://arxiv.org/abs/1801.08624
- Simplifly the model
- Early Stopping
- Cross Validation
- Dropout (NN)
- Use Transfer Learing!
- ...

Reference
---

Colab Notebook:
- https://github.com/huyhoang17/Colab_Temporary/blob/master/Training_CTC_Vietnamese_Recognition_10epochs.ipynb

Papers
- https://arxiv.org/pdf/1804.01527.pdf
- https://arxiv.org/abs/1801.08624
- https://arxiv.org/abs/1308.0850
	- Blog: https://greydanus.github.io/2016/08/21/handwriting/
	- https://github.com/tensorflow/magenta/blob/master/magenta/reviews/summary_generation_sequences.md
	- https://github.com/sjvasquez/handwriting-synthesis
	- https://github.com/greydanus/scribe

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

Overfitting
- https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42
- https://elitedatascience.com/overfitting-in-machine-learning