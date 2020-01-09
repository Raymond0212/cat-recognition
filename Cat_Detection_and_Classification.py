#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers as opt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, MaxPooling1D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


# In[ ]:


SPLIT = 0.2
EPOCHS = 30 #10
IMG_HEIGHT = 116
IMG_WIDTH = 116
BATCH_SIZE = 128


# In[ ]:


path = ".\\img_sub"
path_dog = os.path.join(path, 'cat')
path_nodog = os.path.join(path, 'no_cat')
class_names = ['cat', 'no_cat']

total_img_num = 0

for a in os.listdir(path_dog):
    total_img_num += len(os.listdir(os.path.join(path_dog, a)))

for a in os.listdir(path_nodog):
    total_img_num += len(os.listdir(os.path.join(path_nodog, a)))

total_val = total_img_num * SPLIT
total_train = total_img_num - total_val

print('total images:', total_img_num)
print('number of training data:', total_train)
print('number of validation:', total_val)


# In[ ]:


data_generator = ImageDataGenerator(
    rescale=1./255,rotation_range=40, # Angle, 0-180
    # width_shift_range=0.2, # horizontal shifting
    # height_shift_range=0.2, # vertical shifting
    # shear_range=0.2, # Shearing
    # zoom_range=0.2, # Zooming
    # horizontal_flip=True, # Flipping
    validation_split=SPLIT
)


# In[ ]:


train_generator = data_generator.flow_from_directory(
    directory=path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    BATCH_SIZE=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training') # set as training data

validation_generator = data_generator.flow_from_directory(
    directory=path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    BATCH_SIZE=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='validation') # set as validation data

print(train_generator.class_indices)
# In[ ]:


Detction = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])


# In[ ]:


sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
rms = RMSprop(learning_rate=0.0001, decay=1e-6)


# In[ ]:


Detction.compile(optimizer= sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Detction.summary()  # model summary


# In[ ]:


history = Detction.fit_generator(
    train_generator,
    steps_per_epoch=total_train // BATCH_SIZE,
    EPOCHS=EPOCHS,
    validation_data=validation_generator,
    validation_steps=total_val // BATCH_SIZE
)


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='lower left')
plt.title('Training and Validation Loss')

plt.show()


# In[ ]:


Detction.save('Detection.h5')
Detction.save_weights('Detection_weights.h5')


# In[ ]:


path = ".\\img\\cat"
dirlist = os.listdir(path)
catPath = [os.path.join(path, dirname) for dirname in dirlist]

# In[ ]:
total_img_num = sum(map(len,map(os.listdir, catPath)))
print('all kinds of cats: ', dirlist)
for i in range(len(catPath)):
    print(dirlist[i], ': ', len(os.listdir(catPath[i])))
print('total cat imgs: ', total_img_num)

# ## Train test SPLIT
# In[ ]:
BATCH_SIZE = 10
EPOCHS = 60
IMG_HEIGHT = 116
IMG_WIDTH = 116
SPLIT = 0.2
classNum = len(dirlist)
total_val = total_img_num * SPLIT
total_train = total_img_num * (1 - SPLIT)


# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1./255,rotation_range=40, # Angle, 0-180
    width_shift_range=0.2, # horizontal shifting
    height_shift_range=0.2, # vertical shifting
    shear_range=0.2, # Shearing
    zoom_range=0.2, # Zooming
    horizontal_flip=True, # Flipping
    validation_split=SPLIT
)

# In[ ]:
# Load images from the disk, applies rescaling, and resizes the images
train_generator = train_datagen.flow_from_directory(
    directory=path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    BATCH_SIZE=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
) # set as training data

validation_generator = train_datagen.flow_from_directory(
    directory=path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    BATCH_SIZE=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
) # set as validation data

print(train_generator.class_indices)
# In[ ]:


model = Sequential([
    Conv2D(32, 3, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3), name = 'cov1'),
    Activation('relu', name = 'act1'),
    Conv2D(32, 3, name = 'cov2'),
    Activation('relu', name = 'act2'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool1'),
    Dropout(0.25, name = 'drop1'),
    Conv2D(64, 5, padding='same', name = 'cov3'),
    Activation('relu', name = 'act3'),
    Conv2D(64, 5, name = 'cov4'),
    Activation('relu', name = 'act4'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool2'),
    Dropout(0.25, name = 'drop2'),
    Flatten(name = 'flatten'),
    Dense(512, name = 'dense1'),
    Activation('relu', name = 'act5'),
    Dropout(0.5, name = 'drop3'),
    Dense(classNum, name = 'dense2'),
    Activation('softmax', name = 'act6')
])


# In[ ]:


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
rms = RMSprop(learning_rate=0.0001, decay=1e-6)

model.compile(optimizer=sgd,
             loss='categorical_crossentropy',
             metrics=['accuracy'],  
             )

model.summary()


# In[ ]:


with tf.device('/gpu:0'):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = total_train // BATCH_SIZE,
        EPOCHS = EPOCHS,
        validation_data = validation_generator,
        validation_steps = total_val // BATCH_SIZE     
    )


# In[ ]:


Detction.save('Identification.h5')
Detction.sabve_weights('Identification_weights.h5')

