import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.simplefilter("ignore")

import cv2
import time
import numpy as np

import collections
import traceback
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

from keypoints_extractor import KeypointsExtractor
from detectors_keras_api.pose_detector_keras import PoseDetector
from detectors_keras_api.right_hand_detector_keras import RightHandUpDetector
from detectors_keras_api.initial_pose_detector_keras import InitialPoseDetector
from detectors_keras_api.predict_lstm_model_keras import predict_sequence

from list_manipulator import pop_all
from image_manipulator import crop_image_based_on_padded_bounded_box
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

IMAGE_DIR = '/home/kevin/Pictures/save_images'

# Exercises ['plank', 'push-up', 'sit-up', 'squat']

def validate_keypoints(keypoints):
    if keypoints[7] == 0:
        print(keypoints)

class PlankHandler():
    def __init__(self):
        self.time_end = None

    def handle(self, prediction, list_of_frames, keypoints, start, end):
        if prediction == 1 and self.time_end is None:
            start = True
            self.time_end = time.time() + 1

        if len(list_of_frames) > 12 and self.time_end < time.time():
            end = True

        if start and not end:
            list_of_frames.append(keypoints)

        return start, end, list_of_frames


plank_handler = PlankHandler()


def handle_exercise(prediction, list_of_frames, keypoints, start, end):
    # If starting position is found and start is True then mark end
    if prediction == 1 and start and len(list_of_frames) > 12:
        end = True
        print("end pose")
    
    # If starting position is found and end is False then mark start
    if (len(list_of_frames) == 1 or not end) and prediction == 1:
        start = True
        if len(list_of_frames) == 1:
            print("restart starting pose")
        else:
            print("starting pose")

        # If the found counter is more than one
        # Delete frames and restart collection
        if len(list_of_frames) >= 1:
            pop_all(list_of_frames)

        # Add frames
        list_of_frames.append(keypoints)

    return start, end, list_of_frames


if __name__ == '__main__':
    # Base paths
    base_path = "/home/kevin/projects/right-hand-up-to-exercise/pushup3_24fps.mp4"
    # base_path = "/home/kevin/projects/right-hand-up-to-exercise/squat.mp4"
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

    # Define x_min, y_min, x_max, y_max
    x_min = -1
    y_min = -1
    x_max = -1
    y_max = -1

    start = False
    end = False
    list_of_frames = []
    list_of_lstm_predictions = []

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
                    t_end = time.time() + 7.5
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

                            # Set exercise type and pose detector
                            exercise_type = str(prediction)
                            pose_detector = PoseDetector(exercise_type)

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
            try:
                list_of_keypoints, image_show = kp_extractor.get_keypoints_and_id_from_img(image_to_process)
            except:
                break

            try: 
                if list_of_keypoints == None:
                    break
                x = list_of_keypoints[0]
                if x['ID'] == target_id:
                    # Transform keypoints list to array
                    keypoints = np.array(x['Keypoints']).flatten()

                    # # Get prediction
                    prediction = pose_detector.predict(np.array([keypoints]))

                    if exercise_type == "plank":
                        start, end, list_of_frames = plank_handler.handle(prediction, list_of_frames, keypoints, start, end)
                    else:
                        # If starting position is found and start is True then mark end
                        if prediction == 1 and start and len(list_of_frames) > 12:
                            end = True
                        
                        # If starting position is found and end is False then mark start
                        if (len(list_of_frames) == 1 or not end) and prediction == 1 and len(list_of_frames) <= 1:
                            start = True

                            # If the found counter is more than one
                            # Delete frames and restart collection
                            if len(list_of_frames) >= 1:
                                pop_all(list_of_frames)

                        # validate_keypoints(keypoints)
                        # Add frames
                        if start:
                            list_of_frames.append(keypoints)

                        # If both start and end was found 
                        # send data to LSTM model and Plotter
                        if start and end:
                            # Send data
                            list_of_lstm_predictions.append(predict_sequence(list_of_frames, exercise_type))

                            # Pop all frames in list
                            pop_all(list_of_frames)

                            # Restart found_counter, start flag and end flag
                            start = True
                            end = False

                            # Add frames
                            list_of_frames.append(keypoints)
                else:
                    # If not target
                    continue
                    
            except Exception as e:
                traceback.print_exc()
                print(end="")

        # Display the stream
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", image_show)
        key = cv2.waitKey(1)

        # Quit
        if key == ord('q'):
            break
    
    print(f'{len(list_of_lstm_predictions)} predictions, results: {list_of_lstm_predictions}')