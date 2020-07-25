# coding: utf-8
# # Object Detection Demo
import numpy as np
import time
import random
import argparse
import tensorflow as tf
from PIL import Image
import cv2

# ## Object detection imports
# Here are the imports from the object detection module.

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# # Model preparation
# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

from setting import *
from utils.collect_label_info2csv import convert_xml2csv

import logging

logger = logging.getLogger("main")
# logger.setLevel(logging.DEBUG)  root logger leval is higher than info
# logger.info('info')
# logger.warning('war')
# logger.error('err')
# logging.StreamHandler(sys.stdout)
# logger.setFileLevel(logging.INFO)

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# image.getdata() is very very slow!!!
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# fast !!!
def load_image_into_numpy_array_fixed(image):
    (im_width, im_height) = image.size

    return np.asarray(image).reshape((im_height, im_width, 3)).astype(np.uint8)


# # Detection
class Detection(object):
    def __init__(self):
        # ## Load a (frozen) Tensorflow model into memory.
        self.image_tensor = None
        self.tensor_dict = {}

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)
        self.__create_image_tensor()

    def close(self):
        self.sess.close()

    def __create_image_tensor(self):
        # test
        # print("input data:", image)
        # Get handles to input and output tensors
        with self.detection_graph.as_default():
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}

            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            # if 'detection_masks' in tensor_dict:
            #     # The following processing is only for single image
            #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            #     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            #         detection_masks, detection_boxes, image.shape[0], image.shape[1])
            #     detection_masks_reframed = tf.cast(
            #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            #     # Follow the convention by adding back the batch dimension
            #     tensor_dict['detection_masks'] = tf.expand_dims(
            #         detection_masks_reframed, 0)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # print(np.expand_dims(image,0))

    def run_inference_for_single_image(self, image):
        # Get handles to input and output tensors
        # Run inference
        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


def load_index2class():
    logger.debug('load classify index.')
    classify_index = {}
    with open(GESTURE_CLASS_INDEX, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip().split(',')
            classify_index[value] = key
    return classify_index


def load_class_index():
    logger.debug('load classify index.')
    classify_index = {}
    with open(GESTURE_CLASS_INDEX, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip().split(',')
            classify_index[key] = int(value)
    return classify_index


def generate_test_data_info(xml_path_dir, test_data_info_path):
    convert_xml2csv(os.path.dirname(xml_path_dir), test_data_info_path)
    logger.debug('generate test detail information csv file.')


def detect_hand_gesture(images, annotations, detail_info, output, test_num):
    ts = time.time()
    class2index = load_class_index()
    index2class = load_index2class()

    logger.debug('convert annotations to detail csv file')
    if not os.path.exists(detail_info):
        generate_test_data_info(annotations, detail_info)
    detail_info_dict = {}
    with open(detail_info, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, _, _, cls, _, _, _, _ = line.strip().split(',')
            if cls == "two":
                cls = 'victory'
            if cls not in class2index:
                continue
            if filename not in detail_info_dict:
                detail_info_dict[filename] = [class2index[cls]]
            else:
                detail_info_dict[filename].append(class2index[cls])

    processing_data_time = time.time() - ts
    logger.debug('Processing test data cost {}ms.'.format(processing_data_time * 1000))
    # define TP,FP,Precision; Precision = TP / (TP + FP)
    true_positive = 0
    false_positive = 0

    test_image_paths = [os.path.join(images, image) for image in os.listdir(images)]

    image_nums = len(test_image_paths)
    if image_nums <= 0:
        return False
    if test_num <= 0 or test_num > image_nums:
        test_num = image_nums
        logger.warning('Test number is incorrect, reset it to all test images number!')
    # get shuffle image paths
    index_list = list(range(image_nums))
    random.shuffle(index_list)
    # shuffle_test_image_list = list(map(test_image_paths.__getitem__, index_list))
    shuffle_test_image_list = test_image_paths

    test_image_paths = shuffle_test_image_list[:test_num + 1]

    result_info = ''
    detect_image_num = 0
    detect_boxes_num = 0

    detect_obj = Detection()

    # visualize initial
    # plt.figure(figsize=IMAGE_SIZE)

    for image_path in test_image_paths:
        if not os.path.exists(image_path):
            logger.warning("***Image not exists: {} ***".format(image_path))
            continue

        per_image_start = time.time()
        # Load image
        # keep a consistent pre_processing and inference
        image = Image.open(image_path)
        image_filename = os.path.basename(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array_fixed(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        # object detect
        detect_ts = time.time()
        output_dict = detect_obj.run_inference_for_single_image(image_np)
        detect_time = time.time() - detect_ts

        if image_filename not in detail_info_dict:
            logger.warning("******Image not in detail_info_dict: {}******".format(image_filename))
        try:
            ground_list = detail_info_dict[image_filename]
        except Exception as e:
            continue

        detect_image_num += 1
        boxes_num = len(ground_list)
        detect_list = output_dict['detection_classes'][:boxes_num]
        # ndarray to list
        detect_list.tolist()
        matches = [index for index, elem in enumerate(detect_list)
                   if elem == ground_list[index]]
        true_positive += len(matches)
        false_positive += boxes_num - len(matches)

        detect_boxes_num += boxes_num

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        '''
        # 1. plt.savefig()
        # clear figure
        plt.cla()
        plt.imshow(image_np)
        result_image = os.path.join(output, image_filename)
        plt.savefig(result_image)
        '''

        """
        # 2. Image.save() 
        im = Image.fromarray(image_np)
        im.save(result_image)
        """

        """
        # 3. cv2.imwrite()  is fastest
        """
        result_image = os.path.join(output, image_filename)
        resized = cv2.resize(image_np, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(result_image, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        detect_class_list = [index2class[str(index)] for index in detect_list]

        result_info += '{} detect result:{}\n'.format(image_filename, detect_class_list)

        per_image_end = time.time()

        logger.warning('***Current image: {}/{}, TP:{}, FP:{}. Detect:{:.2f}ms, Per image:{:.2f}ms.'.format(detect_image_num,
                                                                                image_nums, true_positive, false_positive,
                                                                                detect_time * 1000, (per_image_end - per_image_start) * 1000))


    precision = true_positive / (true_positive + false_positive)
    result_info += 'Test {} images, {} boxes, TP: {}, FP: {}, precision: {:.2f}.\n'.format(detect_image_num,
                                                                                     detect_boxes_num,
                                                                                     true_positive,
                                                                                     false_positive,
                                                                                     precision)
    logger.warning(result_info)
    format_filename = 'result-{}.txt'.format(time.strftime("%Y-%m-%d", time.localtime()))
    with open(os.path.join(output, format_filename), 'w', encoding='utf-8') as w:
        w.write(result_info)

    total_time = time.time() - ts
    logger.error('Testing cost {:.4f}s, then finished!'.format(total_time))

    detect_obj.close()

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test hand gesture detect.')
    parser.add_argument('--test_data_dir', type=str,
                        help='test data directory path.')
    parser.add_argument('--test_num', type=int, default=10,
                        help='option test number, default value is 10.')
    parser.add_argument('--output', type=str,
                        help='Testing result dir.')
    args = parser.parse_args()

    test_data_dir = args.test_data_dir

    annotation_dir_path = os.path.join(test_data_dir, 'Annotations')
    jpeg_dir_path = os.path.join(test_data_dir, 'JPEGImages')

    data_detail = os.path.join(test_data_dir, 'data_detail.csv')

    test_state = detect_hand_gesture(jpeg_dir_path, annotation_dir_path, data_detail, args.output, args.test_num)
    if test_state:
        logger.warning('Testing successfully!')
    else:
        logger.error('Testing error!')
    # jpeg_dir_path = 'data/qatest/JPEGImages'
    # annotation_dir_path = 'data/qatest/Annotations'
    # data_detail = 'data/qatest/data_detail.csv'
    # output = 'data/qatest/result'
    # test_num = 2
    #
    # test_state = detect_hand_gesture(jpeg_dir_path, annotation_dir_path, data_detail, output, test_num)


