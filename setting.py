import os
# What model to download.
MODEL_NAME = 'model/hand_gesture_detection'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('utils', 'gesture_label_map.pbtxt')

NUM_CLASSES = 26

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
SAVE_IMAGE_SIZE = (400, 300)

GESTURE_CLASS_INDEX = 'utils/gesture_label_map.txt'
#GESTURE_CLASS_INDEX = 'utils/gesture_label_map.txt'
