import os
import numpy as np
import urllib.parse
from pymongo import MongoClient
import sys
 
import cv2

ROOT_DIR = "/home/kevin/projects/exercise_pose_evaluation_machine"
sys.path.append(ROOT_DIR)

from db_entity import insert_np_array_to_db
from keypoints_extractor import KeypointsExtractor, make_min_max_scaler, normalize_keypoints_from_external_scaler


if __name__ == '__main__':
    # Keypoint list for each exercise
    # KP ordering of body parts
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

    # Initialize dataset path
    dataset_path = '/home/kevin/projects/dataset-theo/to-insert-to-db'

    # Define keypoints
    kp_extractor = KeypointsExtractor()
    
    # Get all keypoints from dataset
    # and insert into database
    dirs = os.listdir(dataset_path)
    for foldername in dirs:
        # Get all folder listz
        folder = dataset_path + '/' + foldername
        files = os.listdir(folder)

        # Repeat for every files
        for filename in files:
            try:
              keypoints = []
              class_type = 0
              print("Processing file: " + filename)

              image_to_process = cv2.imread(filename)
              list_of_keypoints, image_show = kp_extractor.get_upper_body_keypoints_and_id_from_img(image_to_process)


              cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", image_show)
            except Exception as e:
              # Break at end of frame
              break