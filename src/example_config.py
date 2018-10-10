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

# if K.image_data_format() == 'channels_first':
#     INPUT_SHAPE = (NO_CHANNELS, IMG_W, IMG_H)
# else:
#     INPUT_SHAPE = (IMG_W, IMG_H, NO_CHANNELS)
INPUT_SHAPE = (IMG_W, IMG_H, NO_CHANNELS)
IMG_BG_TEXT = ("black", "white")

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
BASE_DATA = ""
SAMPLES_DATA = ""
RAW_DATA = ""
PP_DATA = ""
GEN_DATA = ""
TRANSCRIPTION = ""
TRANSGEN = ""

# naming
