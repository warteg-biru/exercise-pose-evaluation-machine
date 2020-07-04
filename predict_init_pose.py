import os

import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras

import sys
from sys import platform
import numpy as np
import os
import cv2
import tensorflow as tf
import collections
import matplotlib.pyplot as plt
import sys
from sys import platform
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from keypoints_extractor import scan_image


if __name__ == '__main__':
    # Base paths
    checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/"
    img_test_path = "/home/kevin/projects/Exercise Starting Pose/Dumbell Curl/dumbellcurl1.jpg"

    with tf.Session() as session:
        # Load checkpoint model
        loader = tf.train.import_meta_graph(checkpoint_path+'initial_pose_model.ckpt.meta')
        loader.restore(session, tf.train.latest_checkpoint(checkpoint_path))

        # Get required tensors
        predict = session.graph.get_tensor_by_name('predict:0')

        # Get placeholder and keypoint data
        tf_data = session.graph.get_tensor_by_name('tf_data:0')
        keypoints = np.array(scan_image(img_test_path)).flatten().astype(np.float32)

        # Get prediction
        predictions = session.run(predict, feed_dict={tf_data: [keypoints]})

        # Print prediction
        print("-----")
        print(np.argmax(predictions))      
        print("-----")