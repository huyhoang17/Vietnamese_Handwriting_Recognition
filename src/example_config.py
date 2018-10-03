import os

from keras import backend as K


# constans
CHARS = '\ !%"#&\'()*+,-./0123456789:;?AÁẢÀÃẠÂẤẨẦẪẬĂẮẲẰẴẶBCDĐEÉẺÈẼẸÊẾỂỀỄỆFGHIÍỈÌĨỊJKLMNOÓỎÒÕỌÔỐỔỒỖỘƠỚỞỜỠỢPQRSTUÚỦÙŨỤƯỨỬỪỮỰVWXYÝỶỲỸỴZaáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệfghiíỉìĩịjklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvwxyýỷỳỹỵz'  # noqa
CHARS_ = [char for char in CHARS]
PIXEL_INDEX = 127
NO_GEN_IMAGES = 2**5

# sample params
TRAIN_SIZE = 0.95
MAX_LEN_TEXT = 256
IMAGE_SIZE = (1150, 32)
IMG_W, IMG_H = IMAGE_SIZE
NO_CHANNELS = 1

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (NO_CHANNELS, IMG_W, IMG_H)
else:
    INPUT_SHAPE = (IMG_W, IMG_H, NO_CHANNELS)

# model params
NO_EPOCHS = 25
NO_LABELS = 216
BATCH_SIZE = 16
CONV_FILTERS = 16
KERNEL_SIZE = (3, 3)
POOL_SIZE = 2
DOWNSAMPLE_FACTOR = POOL_SIZE ** 2
TIME_DENSE_SIZE = 256
RNN_SIZE = 256

# paths
SAMPLES = "path-to-samples-data-dir"
BASE_DATA = "../data"
RAW_DATA = os.path.join(BASE_DATA, "vn_handwriting_data")
PP_DATA = os.path.join(BASE_DATA, "pp_vn_handwriting_data")
GEN_DATA = os.path.join(BASE_DATA, "gen_data")
LABELS = os.path.join(BASE_DATA, 'path-to-transcription.pk')

# naming
