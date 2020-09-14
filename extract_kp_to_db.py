import os
import numpy as np
import urllib.parse
from pymongo import MongoClient

from db_entity import insert_np_array_to_db
from keypoints_extractor import KeypointsExtractor, make_min_max_scaler, normalize_keypoints_from_external_scaler

'''
set_params

# Set the openpose parameters
'''
# Set openpose default parameters
def set_params():
    '''
    params untuk menambah performance
    '''
    params = dict()
    params["net_resolution"] = "320x176"
    params["face_net_resolution"] = "320x320"
    params["model_pose"] = "BODY_25"
    
    # Ensure you point to the correct path where models are located
    params["model_folder"] = './openpose/models'
    return params

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

'''
validate_keypoints

# Validate keypoints
@params{list of frames} Array of frames
@params{integer} Expected number of frames
'''
def validate_keypoints(frames, expected_frames):
    # If the actual frame count does not 
    # Equal to the expected frame count
    if len(frames) != expected_frames:
        print("Data invalid! Expected " +str(expected_frames) + " frames, received " + str(len(frames)) + ".")
        return False
    for keypoints in frames:
        # If the actual keypoints does not 
        # Equal to the expected keypoints
        if len(keypoints) != 14:
            print("Data invalid! Expected 14 keypoints, received " + str(len(keypoints)) + ".")
            return False
        for coordinates in keypoints:
            # If the actual coordinates does not 
            # Equal to the expected coordinates (x,y)
            if len(coordinates) != 2:
                print("Data invalid! Expected 2 coordinates, received " + str(len(coordinates)) + ".")
                return False
    return True

if __name__ == '__main__':
    # Keypoint list for each exercise
    selected_keypoints = [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE]

    # Initialize dataset path
    dataset_path = '/home/kevin/projects/dataset-theo/Cut/to-process'

    # Define keypoints
    kp_extractor = KeypointsExtractor()
    
    # Get all keypoints from dataset
    # and insert into database
    dirs = os.listdir(dataset_path)
    for foldername in dirs:
        # Get all folder list
        folder = dataset_path + '/' + foldername
        files = os.listdir(folder)

        # Repeat for every files
        for filename in files:
            try:
                keypoints = []
                class_type = 0
                print("Processing file: " + filename)

                # Generate exercise for each exercise type
                if foldername == "push-up":
                    all_exercise_reps, all_exercise_x_low, all_exercise_y_low = kp_extractor.scan_video_without_normalize(folder + '/' + filename, selected_keypoints)
                    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))

                    normalized_reps = normalize_keypoints_from_external_scaler(all_exercise_reps, scaler)
                    if not validate_keypoints(normalized_reps, 24):
                        continue
                    class_type = 1
                elif foldername == "plank":
                    all_exercise_reps, all_exercise_x_low, all_exercise_y_low = kp_extractor.scan_video_without_normalize(folder + '/' + filename, selected_keypoints)
                    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))

                    normalized_reps = normalize_keypoints_from_external_scaler(all_exercise_reps, scaler)
                    if not validate_keypoints(normalized_reps, 24):
                        continue
                    class_type = 2
                elif foldername == "sit-up":
                    all_exercise_reps, all_exercise_x_low, all_exercise_y_low = kp_extractor.scan_video_without_normalize(folder + '/' + filename, selected_keypoints)
                    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))

                    normalized_reps = normalize_keypoints_from_external_scaler(all_exercise_reps, scaler)
                    if not validate_keypoints(normalized_reps, 48):
                        continue
                    class_type = 3

                # Insert keypoints to mongodb
                insert_np_array_to_db(normalized_reps, class_type, foldername)
            except Exception as e:
                print(e)