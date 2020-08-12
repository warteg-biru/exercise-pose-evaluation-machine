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
from right_hand_detector_keras import RightHandUpDetector
from initial_pose_detector_keras import InitialPoseDetector


if __name__ == '__main__':
    # Base paths
    base_path = "/home/kevin/projects/dataset-handsup-to-exercise/pushup.mp4.mp4"
    kp_extractor = KeypointsExtractor()

    # Opening OpenCV stream
    stream = cv2.VideoCapture(base_path)
    
    # Define flags
    target_detected_flag = False
    init_pose_detected = False
    target_id = -1
    
    # Define end time variable 
    # Used as a benchmark for functions
    t_end = time.time()

    # Define detectors
    right_hand_up_detector = RightHandUpDetector()
    initial_pose_detector = InitialPoseDetector()

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
                    # Transform keypoints list to (1, 16) matrix
                    keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                    keypoints = np.array([keypoints])

                    # Get prediction
                    try:
                        prediction = right_hand_up_detector.predict(keypoints)
                        print('right hand up detector predicted -> ', prediction)
                    except Exception as e:
                        print(e)

                    if prediction == 1:
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
            try: 
                for x in list_of_keypoints:
                    if x['ID'] == target_id and t_end <= time.time():
                        # Transform keypoints list to array
                        keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                        keypoints = np.array([keypoints])
                        # Get prediction
                        prediction = initial_pose_detector.predict(keypoints)
                        print("Initital pose prediction result: " + prediction)
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