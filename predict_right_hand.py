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
from keypoints_extractor import scan_image, get_upper_body_keypoints_and_id, get_upper_body_keypoints_and_id_from_img, set_params

from keypoints_extractor import KeypointsExtractor

if __name__ == '__main__':
    # Base paths
    checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up/"
    base_path = "/home/kevin/projects/dataset-handsup-to-exercise/pushup.mp4.mp4"
    kp_extractor = KeypointsExtractor()
    
    with tf.Session() as session:
        # Load checkpoint model
        loader = tf.train.import_meta_graph(checkpoint_path+'right_hand_up.ckpt.meta')
        loader.restore(session, tf.train.latest_checkpoint(checkpoint_path))

        # Get required tensors
        predict = session.graph.get_tensor_by_name('predict:0')

        # Get placeholder and keypoint data
        tf_data = session.graph.get_tensor_by_name('tf_data:0')

        # Opening OpenCV stream
        stream = cv2.VideoCapture(base_path)
        
        while True:
            # Initialize image to process variable
            imageToProcess = ""

            try:
                ret, imageToProcess = stream.read()
            except Exception as e:
                # Break at end of frame
                break

            # Get keypoint and ID data
            list_of_keypoints = kp_extractor.get_upper_body_keypoints_and_id_from_img(imageToProcess)

            # Foreach keypoint predict user data
            found_counter = 0
            found_id = None

            try: 
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
                    break
            except Exception as e:
                # Break at end of frame
                pass
            
            # Display the stream
            cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", imageToProcess)
            key = cv2.waitKey(1)

            # Quit
            if key == ord('q'):
                break