import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from skimage.io import imread
from skimage.transform import resize


from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from google.protobuf import text_format as pbtf
from tensorflow.core.framework import graph_pb2 as gpb

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_output(graph):
    with graph.as_default():
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    return tensor_dict, image_tensor


def run_inference_for_single_image(image, tensor_dict, image_tensor):
    with tf.Session() as sess:
        # Run inference
        output_dict = sess.run(tensor_dict,
                                feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


MODEL_NAME = './models/inference_model_retinanet'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('dataset', 'label_map_obj.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


PATH_TO_TEST_IMAGES_DIR = './dataset/test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, f) for f in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
FIG_SIZE = (12, 8)
IMAGE_SIZE = (760, 480)

import csv
import time

# test_num = 3
# count = 0
with open('./detection.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['image_id', 'labels'])

    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            for image_path in TEST_IMAGE_PATHS:
                print(image_path)
                t = time.time()
                image = Image.open(image_path)
                width, height = image.size
                image = image.resize(IMAGE_SIZE)
                print("load time: ", time.time()-t)
                t = time.time()

                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: image_np_expanded})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                # output_dict = run_inference_for_single_image(image_np_expanded, tensor_dict, image_tensor)
                image_id = os.path.basename(image_path)[:-4]
                labels = ""
                print("inference time: ", time.time()-t)
                t = time.time()
                # print(output_dict['detection_scores'], output_dict['detection_scores'].shape)
                for i in range(output_dict['detection_scores'].shape[0]):
                    if output_dict['detection_scores'][i] > 0.5:
                        # if i != 0:
                        #     labels += ' '
                        # c = category_index[output_dict['detection_classes'][i]]
                        labels = labels + ' obj'
                        bbx = output_dict['detection_boxes'][i]
                        labels = labels + ' ' + str(int(bbx[1]*width))
                        labels = labels + ' ' + str(int(bbx[0]*height))
                        labels = labels + ' ' + str(int(bbx[3]*width))
                        labels = labels + ' ' + str(int(bbx[2]*height))
                        # print(c, bbx)
                labels = labels.strip()
                writer.writerow([image_id, labels])

                # count += 1
                # if count == test_num:
                #     break

