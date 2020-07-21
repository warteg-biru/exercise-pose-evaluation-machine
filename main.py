import os
import time

import cv2
import traceback
import numpy as np

import collections
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import sys
from sys import platform

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.tools.generate_detections import create_box_encoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor

if __name__ == '__main__':
    # Base paths
    right_hand_up_checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up/"
    init_pose_checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/"
    base_path = "/home/kevin/projects/dataset-handsup-to-exercise/pushup.mp4.mp4"
    kp_extractor = KeypointsExtractor()

    with tf.Session() as session:
        with tf.name_scope("hands_up") as hands_up_scope:
            loader = tf.train.import_meta_graph(right_hand_up_checkpoint_path+'right_hand_up.ckpt.meta')
        with tf.name_scope("init_pose") as init_pose_scope:
            init_pose_loader = tf.train.import_meta_graph(init_pose_checkpoint_path+'initial_pose_model.ckpt.meta')

        # Load right hand up checkpoint model ----------
        loader.restore(session, tf.train.latest_checkpoint(right_hand_up_checkpoint_path))
        # - Get required tensors
        predict = session.graph.get_tensor_by_name(hands_up_scope + 'predict:0')
        # - Get placeholder and keypoint data
        tf_data = session.graph.get_tensor_by_name(hands_up_scope + 'tf_data:0')

        # Load init pose checkpoint model ---------------
        init_pose_loader.restore(session, tf.train.latest_checkpoint(init_pose_checkpoint_path))
        # - Get required tensors
        init_pose_predict = session.graph.get_tensor_by_name(init_pose_scope + 'predict:0')

        # Opening OpenCV stream
        stream = cv2.VideoCapture(base_path)
        
        target_detected_flag = False
        init_pose_detected = False
        target_id = -1
        
        t_end = time.time()

        while True:
            try:
                ret, imageToProcess = stream.read()
            except Exception as e:
                # Break at end of frame
                break
            
            # Foreach keypoint predict user data
            found_counter = 0
            found_id = None

            if target_detected_flag == False:
                # Get keypoint and ID data
                list_of_keypoints = kp_extractor.get_upper_body_keypoints_and_id_from_img(imageToProcess)
                
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
                            pass

                    if found_counter == 1:
                        print("Person " + str(found_id) + " raised their hand")
                        target_detected_flag = True
                        target_id = found_id
                        t_end = time.time() + 10
                except Exception as e:
                    # Break at end of frame
                    pass
            else:
                # Get keypoint and ID data
                list_of_keypoints = kp_extractor.get_keypoints_and_id_from_img(imageToProcess)

                # - Get placeholder and keypoint data
                init_tf_data = session.graph.get_tensor_by_name(init_pose_scope + 'tf_data:0')
                try: 
                    for x in list_of_keypoints:
                        if x['ID'] == target_id and t_end <= time.time():
                            # Transform keypoints list to array
                            keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                            
                            # Get prediction
                            predictions = session.run(init_pose_predict, feed_dict={init_tf_data: [keypoints]})
                            print("Initital pose prediction result: " + str(np.argmax(predictions)))
                            init_pose_detected = True
                        else:
                            # If not target
                            pass
                        
                except Exception as e:
                    # Break at end of frame
                    traceback.print_exc()
                    print(e)
                    pass
                
            # Display the stream
            cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", imageToProcess)
            key = cv2.waitKey(1)

            # Quit
            if key == ord('q') or init_pose_detected:
                break