import urllib.parse
from pymongo import MongoClient
from data_preproccess.plot_body_part_positions import plot_body_part_positions

import numpy as np
import matplotlib.pyplot as plt
from list_manipulator import pop_all
from db_entity import get_dataset, insert_threshold_to_db

# KP ordering of body parts
NECK        = {'name': 'NECK', 'value': 1 }
R_SHOULDER  = {'name': 'R_SHOULDER', 'value': 2 }
R_ELBOW     = {'name': 'R_ELBOW', 'value': 3 }
R_WRIST     = {'name': 'R_WRIST', 'value': 4 }
L_SHOULDER  = {'name': 'L_SHOULDER', 'value': 5 }
L_ELBOW     = {'name': 'L_ELBOW', 'value': 6 }
L_WRIST     = {'name': 'L_WRIST', 'value': 7 }
MID_HIP     = {'name': 'MID_HIP', 'value': 8 }
R_HIP       = {'name': 'R_HIP', 'value': 9 }
R_KNEE      = {'name': 'R_KNEE', 'value': 10 }
R_ANKLE     = {'name': 'R_ANKLE', 'value': 11 }
L_HIP       = {'name': 'L_HIP', 'value': 12 }
L_KNEE      = {'name': 'L_KNEE', 'value': 13 }
L_ANKLE     = {'name': 'L_ANKLE', 'value': 14 }

# Class types
CLASS_TYPE = [
    "dumbell-curl",
    "push-up",
    "sit-up",
    # "squat",
    "plank"
]

# Keypoint types
KEYPOINT_TYPE = [
    NECK, R_SHOULDER, R_ELBOW, 
    R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, 
    MID_HIP, R_HIP, R_KNEE, R_ANKLE, 
    L_HIP, L_KNEE, L_ANKLE
]

def get_max_threshold(list_of_poses, body_part_index):
    # Initialize plot list differences for each frame
    movements_x = []
    movements_y = []

    # nrows is to define the number of test data
    # ncols is to define the number plots for each coordinates (x & y)
    for i, pose_frames in enumerate(list_of_poses[:5]):
        for idx, action in enumerate(pose_frames):
            if idx < len(pose_frames) - 1:
                # Get the plotted movements between the two keypoints
                # For each movement at the x axis in body_part_index keypoints add to movements_x
                movements_x.append(abs(pose_frames[idx+1][body_part_index][0] - pose_frames[idx][body_part_index][0]))
                # For each movement at the y axis in body_part_index keypoints add to movements_y
                movements_y.append(abs(pose_frames[idx+1][body_part_index][1] - pose_frames[idx][body_part_index][1]))
    
    # Return maximum movements for (x, y)
    return np.amax(np.array(movements_x)), np.amax(np.array(movements_y))    

def main():
    for class_type in CLASS_TYPE:
        list_of_poses, list_of_labels = get_dataset(class_type)
        for keypoint in KEYPOINT_TYPE:
            x, y = get_max_threshold(list_of_poses, keypoint["value"] - 1)
            insert_threshold_to_db(x, y, class_type, keypoint)
            print("Successfully inserted threshold for " + class_type + ", " + keypoint["name"] + " keypoint: ", x, y)

main()