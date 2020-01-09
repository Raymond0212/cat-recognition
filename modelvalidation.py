import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

detection = load_model(".\\Detection.h5")
detection.load_weights(".\\Detection_weights.h5")
identification = load_model(".\\Identification.h5")
identification.load_weights(".\\Identification_weights.h5")

PATH = ".\\img"

BATCH_SIZE = 128
EPOCHS = 60
IMG_HEIGHT = 116
IMG_WIDTH = 116
IMG_DEPTH = 3

datagen = ImageDataGenerator(
    rescale=1./255,
#     rotation_range=40, # Angle, 0-180
#     width_shift_range=0.2, # horizontNoneal shifting
#     height_shift_range=0.2, # vertical shifting
#     shear_range=0.2, # Shearing
#     zoom_range=0.2, # Zooming
#     horizontal_flip=True, # Flipping
)

# Load images from the disk, applies rescaling, and resizes the images
test_generator = datagen.flow_from_directory(
    directory=PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
) # set as training data
print(test_generator.class_indices)

print('evaluating...')

detection.evaluate_generator(
    test_generator,
    steps = len(test_generator)
)

PATH = os.path.join(PATH, "cat")
dirlist = os.listdir(PATH)
catPath = [os.path.join(PATH, dirname) for dirname in dirlist]

TOTAL_IMG_NUM = sum(map(len,map(os.listdir, catPath)))
print('all kinds of cats: ', dirlist)
for i in range(len(catPath)):
    print(dirlist[i], ': ', len(os.listdir(catPath[i])))
print('total cat imgs: ', TOTAL_IMG_NUM)

BATCH_SIZE = 128
EPOCHS = 60
IMG_HEIGHT = 116
IMG_WIDTH = 116
IMG_DEPTH = 3
SPLIT = 0.2
CLASSNUM = len(dirlist)
TOTAL_CVAL = TOTAL_IMG_NUM * SPLIT
TOTAL_TRAIN = TOTAL_IMG_NUM * (1 - SPLIT)

datagen = ImageDataGenerator(
    rescale=1./255,
#     rotation_range=40, # Angle, 0-180
#     width_shift_range=0.2, # horizontNoneal shifting
#     height_shift_range=0.2, # vertical shifting
#     shear_range=0.2, # Shearing
#     zoom_range=0.2, # Zooming
#     horizontal_flip=True, # Flipping
)

# Load images from the disk, applies rescaling, and resizes the images
test_generator = datagen.flow_from_directory(
    directory=PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
) # set as training data
print(test_generator.class_indices)

identification.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

print('evaluating...')

identification.evaluate_generator(
    test_generator,
    steps = len(test_generator)/2
)