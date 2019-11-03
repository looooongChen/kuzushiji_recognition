"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
"""

import os
import pandas as pd
import csv
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
from PIL import Image
import json
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format

# sz = [760, 480]

def convert_classes(classes, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text

# label_map = {}
# with open('./train.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     firstline = True
#     for row in readCSV:
#         if firstline:
#             firstline = False
#             continue

#         annotations = row[1].split(' ')
#         if len(annotations) < 5:
#             continue
#         for i in range(len(annotations)):
#             if i % 5 == 0:
#                 label = annotations[i]

#                 if label not in label_map:
#                     label_map[label] = len(label_map) + 1

# to_del = []
# for k in label_map.keys():
#     if label_map[k] < 2000:
#         to_del.append(k)
# for del_key in to_del:
#     del label_map[del_key]
# print("classes remianed: ", len(label_map)) 

label_map = {'obj': 1}
items = []

with open('./train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    firstline = True
    for row in readCSV:
        if firstline:
            firstline = False
            continue
        fname = row[0]
        print(fname)
        im = Image.open(os.path.join('./train_images', fname+'.jpg'))
        width, height = im.size
        # im = imread(os.path.join('./train_images', fname+'.jpg'))
        # y_factor, x_factor = sz[0]/im.shape[0], sz[1]/im.shape[1]
        # im = resize(im, sz)
        # im = rescale(im, 0.25)

        # im = imsave(os.path.join('./train_images_rescaled', fname+'.jpg'), im)

        annotations = row[1].split(' ')
        if len(annotations) < 5:
            continue
        for i in range(len(annotations)):
            if i % 5 == 0:
                label, x, y, w, h = annotations[i], int(annotations[i+1]), int(annotations[i+2]), int(annotations[i+3]), int(annotations[i+4])
                # if label in label_map.keys():
                #     items.append([fname+'.jpg', width, height, label, x, y, x+w, y+h])
                # else:
                #     items.append([fname+'.jpg', width, height, 'other', x, y, x+w, y+h])
                items.append([fname+'.jpg', width, height, 'obj', x, y, x+w, y+h])


txt = convert_classes(list(label_map.keys()))
with open('label_map_obj.pbtxt', 'w') as f:
    f.write(txt)

column_name = ['filename', 'width', 'height',
            'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(items, columns=column_name)
xml_df.to_csv("./train_obj.csv", index=None)

