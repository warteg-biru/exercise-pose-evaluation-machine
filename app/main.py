import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.simplefilter("ignore")

import cv2
import time
import numpy as np

import collections
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

from keypoints_extractor import KeypointsExtractor
from detectors_keras_api.pose_detector_keras import PoseDetector
from detectors_keras_api.right_hand_detector_keras import RightHandUpDetector
from detectors_keras_api.initial_pose_detector_keras import InitialPoseDetector

from list_manipulator import pop_all
from image_manipulator import crop_image_based_on_padded_bounded_box
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    pose_detector = PoseDetector("push-up")

    # Define x_min, y_min, x_max, y_max
    x_min = -1
    y_min = -1
    x_max = -1
    y_max = -1

    bbox = []
    while True:
        ret, image_to_process = stream.read()
        if image_to_process is None:
            break
        
        if x_min > -1 and y_min > -1 and x_max > -1 and y_max > -1:
            image_to_process = crop_image_based_on_padded_bounded_box(x_min, y_min, x_max, y_max, image_to_process)

        # Foreach keypoint predict user data
        found_counter = 0
        found_id = None
        found_exercise = None
        image_show = []
        
        list_of_frames = []
        start = False
        end = False

        if target_detected_flag == False:
            # Get keypoint and ID data
            list_of_keypoints, image_show = kp_extractor.get_upper_body_keypoints_and_id_from_img(image_to_process)
            
            try: 
                for x in list_of_keypoints:
                    # Transform keypoints list to (1, 16) matrix
                    keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                    keypoints = np.array([keypoints])

                    # Get prediction
                    try:
                        prediction = right_hand_up_detector.predict(keypoints)
                    except Exception as e:
                        print(end="")

                    if prediction == 1:
                        found_counter+=1
                        found_id = x['ID']
                        

                    if found_counter > 1:
                        print("Too many people raised their hands!")
                        continue
                    
                if found_counter == 1:
                    print("Person " + str(found_id) + " raised their hand")
                    target_detected_flag = True
                    target_id = found_id
                    t_end = time.time() + 8
                    found_counter = 0
            except Exception as e:
                print(end="")

        elif init_pose_detected == False:
            # Get keypoint and ID data
            list_of_keypoints, image_show = kp_extractor.get_keypoints_and_id_from_img(image_to_process)

            try: 
                for x in list_of_keypoints:
                    if x['ID'] == target_id and t_end < time.time():
                        # Transform keypoints list to array
                        keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                        keypoints = np.array([keypoints])

                        # Get prediction
                        prediction = initial_pose_detector.predict(keypoints)
                        if prediction != -1:
                            found_exercise = prediction
                            print("Initial pose prediction result: " + str(prediction))

                            init_pose_detected = True
                            x_min, y_min, x_max, y_max = kp_extractor.get_bounded_coordinates(prediction, image_to_process)
                    else:
                        # If not target
                        if(t_end > time.time()):
                            print(t_end - time.time())
                        continue
                    
            except Exception as e:
                print(end="")
        else:
            # Get keypoint and ID data
            list_of_keypoints, image_show = kp_extractor.get_keypoints_and_id_from_img(image_to_process)

            try: 
                if list_of_keypoints == None:
                    break
                x = list_of_keypoints[0]
                if x['ID'] == target_id:
                    # Transform keypoints list to array
                    keypoints = np.array(x['Keypoints']).flatten()

                    # Get prediction
                    prediction = pose_detector.predict(np.array([keypoints]))

                    # If starting position is found and start is True then mark end
                    if np.argmax(prediction[0]) == 1 and start:
                        end = True
                    
                    # If starting position is found and end is False then mark start
                    if np.argmax(prediction[0]) == 1 and not end:
                        start = True

                        # If the found counter is more than one
                        # Delete frames and restart collection
                        if len(list_of_frames) >= 1:
                            pop_all(list_of_frames)

                    # Add frames
                    list_of_frames.append(keypoints)

                    # If both start and end was found 
                    # send data to LSTM model and Plotter
                    if start and end:
                        # Send data

                        # Pop all frames in list
                        # pop_all(list_of_frames)

                        # Restart found_counter, start flag and end flag
                        start = True
                        end = False

                        # Add frames
                        list_of_frames.append(keypoints)
                else:
                    # If not target
                    continue
                    
            except Exception as e:
                print(end="")

        # Display the stream
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", image_show)
        key = cv2.waitKey(1)

        # Quit
        if key == ord('q'):
            break