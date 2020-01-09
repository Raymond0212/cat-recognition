import tensorflow as tf
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

detection = load_model(".\\Detection.h5")
detection.load_weights(".\\Detection_weights.h5")
identification = load_model(".\\Identification.h5")
identification.load_weights(".\\Identification_weights.h5")

sys.stderr = stderr

IMG_HEIGHT = 116
IMG_WIDTH = 116
IMG_DEPTH = 3
PATH = ".\\prediction\\img.jpg"
ZERO = 1e-5

img = image.img_to_array(image.load_img(
    PATH, 
    target_size=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
))

img = np.expand_dims(img, axis=0)

detection_result = detection.predict(img)

print(detection_result)
print(detection_result.argmax(axis=-1))

if abs(detection_result[0][0]-1) <= ZERO:
    identification_result = identification.predict(img)
    print(identification_result)
    print(identification_result.argmax(axis=-1))
else:
    print("No cats detected")