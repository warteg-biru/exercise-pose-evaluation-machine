import os
import cv2
import sys
import numpy as np
import collections

from tensorflow.keras.models import load_model

# Get the local openpose path
sys.path.append("/home/kevin/projects/exercise_pose_evaluation_machine")

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor

from keypoints_extractor import normalize_keypoints_for_plot_kps, KeypointsExtractor, make_min_max_scaler, normalize_keypoints_from_external_scaler
from db_entity import get_dataset
from list_manipulator import get_exact_frames

def predict_sequence(keypoints, type_name):
    # Base paths
    save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/lstm_model/keras/" + type_name + "/" + type_name + "_lstm_model.h5"

    # Load model
    model = load_model(save_path)

    keypoints = get_exact_frames(keypoints, type_name)

    # print(f"actual result: {model.predict(np.array([keypoints]))}")
    # Get prediction
    return "1" if model.predict(np.array([keypoints])) > 0.8 else "0"
    

if __name__ == f'__main__':
    # Initialize video path
    # positive
    base_path = "/home/kevin/projects/dataset-theo/optimized/sit-up/sit-up1.mp4"
    # negative
    # base_path = '/home/kevin/Desktop/squat-speedup.mp4'

    # Keypoints 
    NOSE        = 0
    NECK        = 1
    R_SHOULDER  = 2
    R_ELBOW     = 3
    R_WRIST     = 4
    L_SHOULDER  = 5 
    L_ELBOW     = 6
    L_WRIST     = 7
    MID_HIP     = 8
    R_HIP       = 9
    R_KNEE      = 10
    R_ANKLE     = 11
    L_HIP       = 12
    L_KNEE      = 13
    L_ANKLE     = 14
    R_EYE       = 15
    L_EYE       = 16
    R_EAR       = 17
    L_EAR       = 18
    L_BIG_TOE   = 19
    L_SMALL_TOE = 20
    L_HEEL      = 21
    R_BIG_TOE   = 22
    R_SMALL_TOE = 23
    R_HEEL      = 24
    selected_keypoints = [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE]

    # Get reps keypoints
    kp_extractor = KeypointsExtractor()
    all_exercise_reps, all_exercise_x_low, all_exercise_y_low = kp_extractor.scan_video_without_normalize(base_path, selected_keypoints)
    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))
    normalized_reps = normalize_keypoints_from_external_scaler(all_exercise_reps, scaler)
    reshaped_normalized_reps = [np.array(frames).flatten() for frames in normalized_reps]

    # Print results
    print("Result is " + predict_sequence(reshaped_normalized_reps, "sit-up"))
    # JAPHNE MADAFAKIN DID IT AGAIN!
    # JAPHNE MADAFAKIN DID IT AGAIN!
    # JAPHNE MADAFAKIN DID IT AGAIN!
    # JAPHNE MADAFAKIN DID IT AGAIN!