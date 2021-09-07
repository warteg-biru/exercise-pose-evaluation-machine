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

import ffmpeg

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

def write_text(img, text, org, fontScale=.8, thickness=2):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Blue color in BGR 
    color = (255, 0, 255)
    # Using cv2.putText() method 
    return cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

def validate_keypoints(keypoints):
    if keypoints[7] == 0:
        print(keypoints)


VIDEO_PATHS = {
    "push-up": "/home/kevin/projects/dataset/push-up-t.mp4", # OK
    "sit-up": "/home/kevin/projects/dataset/sit-up-t.mp4", # OK
    "squat": "/home/kevin/projects/dataset/squat-k1.mp4", # OK
    "plank": "/home/kevin/projects/dataset/plank-k1.mp4", # OK
    # "test": "/home/kevin/Videos/VID_20210116_152545.mp4"
}


if __name__ == '__main__':
    try:
        # Base paths
        exercise = "sit-up" 
        base_path = VIDEO_PATHS[exercise]
        # base_path = "/home/kevin/projects/dataset/squat-obscured.mp4"
        # base_path = "/home/kevin/projects/right-hand-up-to-exercise/squat.mp4"
        kp_extractor = KeypointsExtractor()

        # Opening OpenCV stream
        stream = cv2.VideoCapture(base_path)
        fps = stream.get(cv2.CAP_PROP_FPS)
        
        # Define flags
        init_pose_detected = False
        target_id = -1

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
        correct_reps = 0
        list_of_processed_frames = []

        bbox = []
        while True:
            ret, image_to_process = stream.read()
            if image_to_process is None:
                break
            
            # if x_min > -1 and y_min > -1 and x_max > -1 and y_max > -1:
            #     image_to_process = crop_image_based_on_padded_bounded_box(x_min, y_min, x_max, y_max, image_to_process)

            # Foreach keypoint predict user data
            found_counter = 0
            found_id = None
            found_exercise = None
            image_show = []

            if init_pose_detected == False:
                # Get keypoint and ID data
                list_of_keypoints, image_show = kp_extractor.get_keypoints_and_id_from_img(image_to_process)

                try: 
                    for x in list_of_keypoints:
                        # Save person ID
                        found_id = x['ID']

                        # Transform keypoints list to array
                        keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                        keypoints = np.array([keypoints])

                        # Get prediction
                        prediction = initial_pose_detector.predict(keypoints)
                        if prediction != -1:
                            found_exercise = prediction
                            target_id = found_id
                            print("Initial pose prediction result: " + str(prediction))

                            # Set exercise type and pose detector
                            exercise_type = str(prediction)
                            pose_detector = PoseDetector(exercise_type)

                            init_pose_detected = True
                            x_min, y_min, x_max, y_max = kp_extractor.get_bounded_coordinates(prediction, image_to_process)
                        
                except Exception as e:
                    print(end="")
            else:
                # Get keypoint and ID data
                try:
                    list_of_keypoints, image_show = kp_extractor.get_keypoints_and_id_from_img(image_to_process)
                    image_show = write_text(image_show, f'type: {exercise_type}', (30, 30))
                    if exercise_type == "plank":
                        image_show = write_text(image_show, f'time: {correct_reps/fps} seconds', (30, 53))
                    else:
                        image_show = write_text(image_show, f'reps: {len(list_of_lstm_predictions)}', (30, 53))
                        image_show = write_text(image_show, f'correct_reps: {correct_reps}', (30, 76))
                except Exception as e:
                    traceback.print_exc()
                    break

                try: 
                    if list_of_keypoints == None:
                        break
                    for x in list_of_keypoints:
                        if x['ID'] == target_id:
                            # Transform keypoints list to array
                            keypoints = np.array(x['Keypoints']).flatten()

                            # Get prediction
                            prediction = pose_detector.predict(np.array([keypoints]))

                            if exercise_type == "plank":
                                if prediction == 1:
                                    correct_reps += 1
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
                                    pred_result = predict_sequence(list_of_frames, exercise_type)
                                    if pred_result == "1":
                                        correct_reps += 1
                                    list_of_lstm_predictions.append(pred_result)

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
            height, width, _= image_show.shape
            list_of_processed_frames.append(cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1)

            # Quit
            if key == ord('q'):
                break
        
        if exercise_type == "plank":
            print(f'{correct_reps/fps} seconds of planks')
        else:
            print(f'{len(list_of_lstm_predictions)} predictions, results: {list_of_lstm_predictions}')

        
        ff_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(exercise +'.mp4', pix_fmt='yuv420p', vcodec='mjpeg', r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        for i in range(len(list_of_processed_frames)):
            ff_proc.stdin.write(list_of_processed_frames[i].astype(np.uint8).tobytes())
        
        ff_proc.stdin.close()

    except:
        traceback.print_exc()
        print(end="")
