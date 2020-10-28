import sys
import os
import cv2
import time
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor
from db_entity import insert_right_hand_up_pose_to_db

from logs.csv_logger import CSVLogger

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# Validate Upper body keypoints. Should have 8 elements
def validate_keypoints(keypoints):
    if len(keypoints) != 8:
        print("Data invalid! Expected 14 keypoints, received " + str(len(keypoints)) + ".")
        return False
    for coordinates in keypoints:
        if len(coordinates) != 2:
            print("Data invalid! Expected 2 coordinates, received " + str(len(coordinates)) + ".")
            return False
    return True


def create_logger():
    date_string = datetime.now().isoformat()
    logger = CSVLogger(
        log_path=os.path.join(
            CURR_DIR, f"logs/kp_extraction_logs/upper_body_{date_string}.csv"
        ),
        headers=["image_res", "time"]
    )
    return logger

def run():
    # Initialize paths
    # TODO: Setup the train data in 2 folders true and false folders
    base_path = "/home/kevin/projects/right_hand_up_pose_data/training_data"

    # Get dataset folders
    dirs = os.listdir(base_path)
    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()
    # Logger
    kp_extraction_logger = create_logger()
    kp_extraction_logger.write_header()

    min_max_scaler = MinMaxScaler(feature_range=(0,1))

    # Label should be either "true" / "false" (Which are the folder names)
    for label in dirs:
        label_dir = os.listdir(base_path+'/'+label)
        for file_name in label_dir:
            file_path = f'{base_path}/{label}/{file_name}'

            image = cv2.imread(file_path)
            # Get image dimensions for logging
            height, width, channels = image.shape
            
            try:
                # record the time to extract keypoints
                kp_extract_start = time.perf_counter()
                keypoints = kp_extractor.get_upper_body_keypoint(file_path)
                kp_extract_end = time.perf_counter()
            except:
                continue

            kp_extraction_log_entry = {
                "image_res": f"{width}x{height}",
                "time": kp_extract_end - kp_extract_start,
            }
            kp_extraction_logger.write_row(kp_extraction_log_entry)
            
            if validate_keypoints(keypoints):
                normalized_keypoints = min_max_scaler.fit_transform(keypoints)
                normalized_keypoints = normalized_keypoints.tolist()
                # print(normalized_keypoints)
                # print(type(normalized_keypoints))
                insert_right_hand_up_pose_to_db(normalized_keypoints, label)


if __name__ == "__main__":
    run()