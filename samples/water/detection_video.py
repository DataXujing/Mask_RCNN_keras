
# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
 import PIL.Image as Image
import cv2
import time
from mrcnn.config import Config
from datetime import datetime 

import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log

from coco import CocoConfig

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

VIDEO_PATH = "./test.mp4"


def skimage2opencv(img):
    src *= 255
    src.astype(int)
    cv2.cvtColor(src,cv2.COLOR_RGB2BGR)
    return src

def opencv2skimage(img):
    cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    src.astype(float32)
    src /= 255
    return src


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
   
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image



class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=DEFAULT_LOGS_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


class_names = ['BG', 'water','Huojian']

file_names = os.listdir(IMAGE_DIR)


# video
vid = cv2.VideoCapture(VIDEO_PATH)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoWriter = cv2.VideoWriter('maskrcnn.mp4', fourcc, video_fps, (video_width, video_height))



for i in range(video_frame_cnt):
    ret, image = vid.read()
    image = cv2.imread(img_path)
    figsize = (image[1],image[0])
    image = skimage2opencv(image)

    start_time = datetime.now() 
    # Run detection
    results = model.detect([image], verbose=1)
    end_time = datetime.now() 
    # Visualize results
    itime = (end_time-start_time).seconds
    print("推断时间：{}".format(itime))
    r = results[0]
    
    fig = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'],figsize=figsize)
    # fig = plt.gcf()
    
    image = fig2data(fig)
    image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    fps = 1000.0 / (itime * 1000)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_BGR, "Mask R-CNN (backbone:ResNet50) | Tesla V100 | FPS: {}".format(fps), (40,40), font, 2, (0,255,0), 3)

    videoWriter.write(image_BGR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
videoWriter.release()

