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

def predict_initial_pose(keypoints):
    # Base paths
    save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5"

    # Prepare keypoints to feed into network
    keypoints = np.array(keypoints).flatten().astype(np.float32)

    # Load model
    model = load_model(save_path)

    # Summarize model
    model.summary()

    # Get prediction
    res_lookup = ['plank', 'sit-up', 'dumbell-curl', 'push-up']
    return res_lookup[np.argmax(model.predict(np.array([keypoints])))]

if __name__ == '__main__':
    img_test_path = "/home/kevin/projects/Exercise Starting Pose/Pushups/pushupstart9.jpg"
    image = cv2.imread(img_test_path)    
    kp_extractor = KeypointsExtractor()
    list_of_pose_and_id = kp_extractor.get_keypoints_and_id_from_img(image)
    keypoints = list_of_pose_and_id[0]['Keypoints']
    print("Result is " + predict_initial_pose(keypoints))