import os
import urllib.parse
from pymongo import MongoClient

from keypoints_extractor import KeypointsExtractor
from db_entity import insert_np_array_to_db

'''
set_params

# Set the openpose parameters
'''
# Set openpose default parameters
def set_params():
    params = dict()
    '''
         params untuk menambah performance
        '''
    params["net_resolution"] = "320x176"
    params["face_net_resolution"] = "320x320"
    params["model_pose"] = "BODY_25"

    # params["logging_level"] = 3
    # params["output_resolution"] = "-1x-1"
    # params["net_resolution"] = "-1x368"
    # params["alpha_pose"] = 0.6
    # params["scale_gap"] = 0.3
    # params["scale_number"] = 1
    # params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    # params["num_gpu_start"] = 0
    # params["disable_blending"] = False
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

def validate_keypoints(frames, expected_frames):
    # If the actual frame count does not 
    # Equal to the expected frame count
    if len(frames) != expected_frames:
        print("Data invalid!")
        return False
    for keypoints in frames:
        # If the actual keypoints does not 
        # Equal to the expected keypoints
        if len(keypoints) != 14:
            print("Data invalid!")
            return False
        for coordinates in keypoints:
            # If the actual coordinates does not 
            # Equal to the expected coordinates (x,y)
            if len(coordinates) != 2:
                print("Data invalid!")
                return False
    return True

if __name__ == '__main__':
    # Keypoint list for each exercise
    selected_keypoints = [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE]

    # Initialize dataset path
    dataset_path = '/home/kevin/projects/dataset-theo/Cut'

    kp_extractor = KeypointsExtractor()
    
    # Get all keypoints from dataset
    # and insert into database
    dirs = os.listdir(dataset_path)
    for foldername in dirs:
        folder = dataset_path + '/' + foldername
        files = os.listdir(folder)
        for filename in files:
            keypoints = []
            class_type = 0
            if foldername == "squat":
                keypoints = kp_extractor.scan_video(folder + '/' + filename, selected_keypoints)
                if not validate_keypoints(keypoints, 48):
                    continue
                class_type = 1
            elif foldername == "push-up":
                keypoints = kp_extractor.scan_video(folder + '/' + filename, selected_keypoints)
                if not validate_keypoints(keypoints, 24):
                    continue
                class_type = 2
            elif foldername == "plank":
                keypoints = kp_extractor.scan_video(folder + '/' + filename, selected_keypoints)
                if not validate_keypoints(keypoints, 24):
                    continue
                class_type = 3
            elif foldername == "sit-up":
                keypoints = kp_extractor.scan_video(folder + '/' + filename, selected_keypoints)
                if not validate_keypoints(keypoints, 48):
                    continue
                class_type = 4
            elif foldername == "dumbell-curl":
                keypoints = kp_extractor.scan_video(folder + '/' + filename, selected_keypoints)
                if not validate_keypoints(keypoints, 24):
                    continue
                class_type = 5

            # Insert keypoints to mongodb
            insert_np_array_to_db(keypoints, class_type, foldername)