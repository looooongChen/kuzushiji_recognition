from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np

import tensorflow as tf
keras = tf.keras
import pandas as pd
import csv
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import json
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import random
from skimage.color import rgb2gray

tf.app.flags.DEFINE_integer("max_class", 10000, "Number of classes")
tf_flags = tf.app.flags.FLAGS

max_class = tf_flags.max_class
channel = 3

label_dict = {}

with open("./dataset/label_counts.txt", "r") as lines:
    for ind, line in enumerate(lines):
        if ind >= max_class:
            break
        parsed = line.strip().split(' ')
        label_dict[ind] = parsed[0]

# load model
# base_model = tf.keras.applications.ResNet50(input_shape=(64, 64, channel),
#                                             include_top=False,
#                                             weights='imagenet')
base_model = tf.keras.applications.Xception(input_shape=(128, 128, 3),
                                            include_top=False,
                                            weights='imagenet')

features = base_model.output
average_pooling = tf.keras.layers.GlobalAveragePooling2D()(features)
predictions = keras.layers.Dense(len(label_dict), activation='softmax')(average_pooling)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.load_weights("./models/model_class{}_xception.h5".format(len(label_dict)))

# prediction
results = open('./submission.csv', 'w', newline='')
results_writer = csv.writer(results)
results_writer.writerow(['image_id', 'labels'])

count = 0
with open('./detection.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    row_num = 0
    for row in readCSV:
        if row_num == 0:
            row_num += 1
            continue
        fname = row[0]
        print(fname)

        im = imread(os.path.join('./dataset/test_images', fname+'.jpg'))

        annotations = row[1].strip().split(' ')
        if len(annotations) < 5:
            results_writer.writerow([fname.strip(), ''])
            continue
            
        coords = []
        patches = []
        for i in range(len(annotations)):
            if i % 5 == 0:
                _, Xmin, Ymin, Xmax, Ymax = annotations[i], int(annotations[i+1]), int(annotations[i+2]), int(annotations[i+3]), int(annotations[i+4])
                patch = resize(im[Ymin:Ymax, Xmin:Xmax], (128, 128), preserve_range=True)
                patch = rgb2gray(patch)
                patch = ((patch-patch.min())/(patch.max()-patch.min())*255).astype(np.uint8)
                patch = np.expand_dims(patch, axis=-1)
                patch = np.concatenate((patch, patch, patch), axis=-1)
                coords.append([int((Xmin+Xmax)/2), int((Ymin+Ymax)/2)])
                patches.append(patch)

        preds = model.predict(np.array(patches))
        labels = ""
        preds = np.argmax(preds, axis=1)
        print('prediction num: ', len(coords))
        for i in range(len(preds)):
            labels += label_dict[int(preds[i])] + ' ' + str(coords[i][0]) + ' ' + str(coords[i][1]) + ' '
        results_writer.writerow([fname.strip(), labels.strip()])

results.close()
