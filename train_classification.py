from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np

import tensorflow as tf
keras = tf.keras
import pandas as pd
import csv
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
from PIL import Image
import json
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from keras.preprocessing.image import ImageDataGenerator
from google.protobuf import text_format
import random
from skimage.color import rgb2gray

tf.app.flags.DEFINE_integer("max_class", 100, "Number of classes")
tf_flags = tf.app.flags.FLAGS

max_class = tf_flags.max_class

label_num = {}
label_counts = {}

with open("./dataset/label_counts.txt", "r") as lines:
    for ind, line in enumerate(lines):
        if ind >= max_class:
            break
        parsed = line.strip().split(' ')
        label_num[parsed[0]] = ind
        label_counts[parsed[0]] = int(parsed[1])
        if ind == 0:
            max_count = int(parsed[1])
# print(label_counts)

train_images = []
train_labels = []

for class_label, class_num in label_num.items():
    print("loading label: ", class_label)
    images = np.load(os.path.join('./dataset/chars', class_label+'.npy'))
    for img in images:
        img = rgb2gray(img)
        img = resize(img, (128, 128))
        img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
        train_images.append(img)
    train_labels = train_labels + [class_num] * len(images)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
ind = np.arange(len(train_labels))
random.shuffle(ind)
train_images = train_images[ind]
train_labels = train_labels[ind]
train_images = np.expand_dims(train_images, axis=-1)
train_images = np.concatenate((train_images, train_images, train_images), axis=-1)

epochs = 80
# fine_tune_at = 100
BATCH_SIZE = 32
base_learning_rate = 0.0001

# base_model = tf.keras.applications.ResNet50(input_shape=(64, 64, 3),
#                                             include_top=False,
#                                             weights='imagenet')
base_model = tf.keras.applications.Xception(input_shape=(128, 128, 3),
                                            include_top=False,
                                            weights='imagenet')


base_model.trainable = True
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable =  False

features = base_model.output
average_pooling = tf.keras.layers.GlobalAveragePooling2D()(features)
average_pooling = tf.keras.layers.Dropout(0.2)(average_pooling)
predictions = keras.layers.Dense(len(label_num), activation='softmax')(average_pooling)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

# datagen = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1)

# model.fit_generator(datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
#                 steps_per_epoch=len(train_images) / BATCH_SIZE, epochs=epochs, use_multiprocessing=False)

for i in range(epochs):
    model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=1)
    model.save("./models/model_class{}_xception.h5".format(len(label_num)))