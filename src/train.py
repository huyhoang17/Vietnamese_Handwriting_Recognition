import pickle

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD

from src.callbacks import VizCallback
import src.config as cf
from src.models import CRNN_model
from src.loaders import TextSequenceGenerator
from src.log import get_logger

logger = get_logger(__name__)


def train():

    with open(cf.TRANSCRIPTION, 'rb') as f:
        data = pickle.load(f)
    no_samples = len(data)
    no_train_set = int(no_samples * cf.TRAIN_SIZE)
    no_val_set = no_samples - no_train_set
    logger.info("No train set: %d", no_train_set)
    logger.info("No val set: %d", no_val_set)

    train_set = TextSequenceGenerator(
        data[:no_train_set],
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=4,
        shuffle=True
    )
    test_set = TextSequenceGenerator(
        data[no_train_set:],
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=4,
        shuffle=False
    )

    model, y_func = CRNN_model()

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    ckp = ModelCheckpoint(
        "path-to-model_cp_CTC_jps_{}epochs.hdf5".format(cf.NO_EPOCHS),
        monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
    )
    vis = VizCallback(
        y_func, test_set, text_size=256, num_display_words=6
    )

    # model = tf.contrib.tpu.keras_to_tpu_model(
    # model,
    # strategy=tf.contrib.tpu.TPUDistributionStrategy(
    #     tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

    model.fit_generator(generator=train_set,
                        steps_per_epoch=no_train_set // cf.BATCH_SIZE,
                        epochs=cf.NO_EPOCHS,
                        validation_data=test_set,
                        validation_steps=no_val_set // cf.BATCH_SIZE,
                        callbacks=[ckp, earlystop, vis])

    return model, y_func
