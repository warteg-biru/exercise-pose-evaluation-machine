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
from keypoints_extractor import scan_image, get_upper_body_keypoints_and_id


if __name__ == '__main__':
    # Base paths
    checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up/"
    checkpoint_path2 = "/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up2/"
    base_path = "/home/kevin/projects/exercise_pose_evaluation_machine/not_important_folder/right-hand-up-test/imagi4.jpg"
    
    with tf.Session() as session:
        # Load checkpoint model
        loader = tf.train.import_meta_graph(checkpoint_path+'right_hand_up.ckpt.meta')
        loader.restore(session, tf.train.latest_checkpoint(checkpoint_path))

        # Get required tensors
        predict = session.graph.get_tensor_by_name('predict:0')

        # Get placeholder and keypoint data
        tf_data = session.graph.get_tensor_by_name('tf_data:0')

        # Get keypoint and ID data
        list_of_keypoints = get_upper_body_keypoints_and_id(base_path)

        # Foreach keypoint predict user data
        found_counter = 0
        found_id = None
        for x in list_of_keypoints:
            # Transform keypoints list to array
            keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)

            # Get prediction
            predictions = session.run(predict, feed_dict={tf_data: [keypoints]})

            # Print prediction
            if predictions[0] < 0.35:
                found_counter+=1
                found_id = x['ID']

            if found_counter > 1:
                print("Too many people raised their hands!")
                break

        if found_counter == 0:
            print("No one was found")
        elif found_counter == 1:
            print("Person " + str(found_id) + " raised their hand")

    # with tf.Session() as session:
    #     # Load checkpoint model
    #     loader = tf.train.import_meta_graph(checkpoint_path2+'right_hand_up.ckpt.meta')
    #     loader.restore(session, tf.train.latest_checkpoint(checkpoint_path2))

    #     # Get required tensors
    #     predict = session.graph.get_tensor_by_name('predicted:0')

    #     # Get placeholder and keypoint data
    #     tf_data = session.graph.get_tensor_by_name('tf_data:0')
        
    #     class_dir = os.listdir(base_path)
    #     for file_name in class_dir:
    #         file_path = f'{base_path}/{file_name}'

    #         keypoints = np.array(get_upper_body_keypoints(file_path)).flatten().astype(np.float32)

    #         # Get prediction
    #         predictions = session.run(predict, feed_dict={tf_data: [keypoints]})

    #         # Print prediction
    #         print("-----")
    #         print(np.argmax(predictions), "supposed to be 0")
    #         print("-----")
        
