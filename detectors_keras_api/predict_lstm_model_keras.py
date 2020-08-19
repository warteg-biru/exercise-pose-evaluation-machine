import os
import cv2
import sys
import numpy as np
import collections

from tensorflow.keras.models import load_model

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor

def predict_sequence(keypoints, type_name):
    # Base paths
    save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/lstm_model/keras/" + type_name + "/" + type_name + "_lstm_model.h5"

    # Load model
    model = load_model(save_path)

    # Summarize model
    model.summary()

    # Get prediction
    res_lookup = ['plank', 'sit-up', 'dumbell-curl', 'push-up']
    return res_lookup[np.argmax(model.predict(np.array([keypoints])))]

if __name__ == '__main__':
    # Initialize video path
    base_path = "/home/kevin/projects/dataset-theo/Cut/push-up/push-up845.mp4"
    
    # Get reps keypoints
    kp_extractor = KeypointsExtractor()
    all_exercise_reps, all_exercise_x_low, all_exercise_y_low = kp_extractor.scan_video_without_normalize(folder + '/' + filename, selected_keypoints)
    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))
    normalized_reps = normalize_keypoints_from_external_scaler(all_exercise_reps, scaler)
    
    # Print results
    print("Result is " + predict_sequence(predict_sequence, "push-up"))