from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

detection = load_model(".\\Detection.h5")
detection.load_weights(".\\Detection_weights.h5")
identification = load_model(".\\Identification.h5")
identification.load_weights(".\\Identification_weights.h5")

split = 0.2
A = 10
epochs = 25 #10
IMG_HEIGHT = 112
IMG_WIDTH = 112
batch_size = 128

path = ".\\prediction"

bi_data_gen = ImageDataGenerator().flow_from_directory(
    directory=path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
)

detection.predict_generator(bi_data_gen)
