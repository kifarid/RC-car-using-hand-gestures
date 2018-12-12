
import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
import uuid # for generating unique file names
import math # math functions

from IPython.display import display as ipydisplay, Image, clear_output, HTML # for interacting with the notebook better

import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))
import matplotlib.pyplot as plt
import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
print('Keras image data format: {}'.format(K.image_data_format()))

CURR_POSE = 'five'
DATA = 'validation_data'


# Begin capturing video
video = cv2.VideoCapture(1)
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame
ok, frame = video.read()
if not ok:
    print("Cannot read video")
    sys.exit()
# Use the first frame as an initial background frame
bg = frame.copy()

# Kernel for erosion and dilation of masks
kernel = np.ones((3, 3), np.uint8)

# Tracking
# Bounding box -> (TopRightX, TopRightY, Width, Height)
bbox_initial = (60, 60, 170, 170)
bbox = bbox_initial

# Text display positions
positions = {
    'hand_pose': (15, 40),
    'fps': (15, 20)
}

# Image count for file name
img_count = 0

# Capture, process, display loop
while True:
    # Read a new frame
    ok, frame = video.read()
    display = frame.copy()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Processing
    # First find the absolute difference between the two images
    diff = cv2.absdiff(bg, frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    # Threshold the mask
    th, thresh = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY_INV)
    thresh_2 = cv2.bitwise_not(thresh)
    # Opening, closing and dilation
    opening = cv2.morphologyEx(thresh_2, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img_dilation = cv2.dilate(closing, kernel, iterations=2)

    # Use numpy array indexing to crop the foreground frame
    hand_crop = img_dilation[ int(bbox[ 1 ]):int(bbox[ 1 ] + bbox[ 3 ]), int(bbox[ 0 ]):int(bbox[ 0 ] + bbox[ 2 ]) ]

    # Draw bounding box
    p1 = (int(bbox[ 0 ]), int(bbox[ 1 ]))
    p2 = (int(bbox[ 0 ] + bbox[ 2 ]), int(bbox[ 1 ] + bbox[ 3 ]))
    cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

    # Display result
    cv2.imshow("display", display)
    # Display diff
    cv2.imshow("diff", diff)
    # Display thresh
    cv2.imshow("thresh", thresh)
    # Display mask
    cv2.imshow("img_dilation", img_dilation)
    try:
        # Display hand_crop
        cv2.imshow("hand_crop", hand_crop)
    except:
        pass

    k = cv2.waitKey(1) & 0xff

    if k == 27:
        break  # ESC pressed
    elif k == 114 or k == 112:
        # r pressed
        bg = frame.copy()
        bbox = bbox_initial

    elif k == 115:
        # s pressed`
        img_count += 1
        fname = os.path.join(DATA, CURR_POSE, "{}_{}.jpg".format(CURR_POSE, img_count))
        cv2.imwrite(fname, hand_crop)
    elif k != 255:
        print(k)

cv2.destroyAllWindows()
video.release()