import sys
import os
import cv2
import time
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor
from db_entity import insert_initial_pose_to_db

from logs.csv_logger import CSVLogger

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def validate_keypoints(keypoints):
    if len(keypoints) != 14:
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
            CURR_DIR, f"logs/kp_extraction_logs/{date_string}.csv"
        ),
        headers=["exercise_type", "image_res", "time"]
    )
    return logger

def run():
    # Initialize paths
    base_path = "/home/kevin/projects/initial-pose-data/train_data"
    save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5"

    # Get dataset folders
    dirs = os.listdir(base_path)
    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()
    # Logger
    kp_extraction_logger = create_logger()
    kp_extraction_logger.write_header()

    min_max_scaler = MinMaxScaler(feature_range=(0,1))

    for class_name in dirs:
        class_dir = os.listdir(base_path+'/'+class_name)
        for file_name in class_dir:
            file_path = f'{base_path}/{class_name}/{file_name}'

            image = cv2.imread(file_path)
            # Get image dimensions for logging
            height, width, channels = image.shape
            
            # record the time to extract keypoints
            kp_extract_start = time.perf_counter()
            list_of_pose_and_id = kp_extractor.get_keypoints_and_id_from_img(image)
            kp_extract_end = time.perf_counter()

            kp_extraction_log_entry = {
                "image_res": f"{width}x{height}",
                "time": kp_extract_end - kp_extract_start,
                "exercise_type": class_name
            }
            kp_extraction_logger.write_row(kp_extraction_log_entry)
            
            keypoints = list_of_pose_and_id[0]['Keypoints']
            if validate_keypoints(keypoints):
                normalized_keypoints = min_max_scaler.fit_transform(keypoints)
                normalized_keypoints = normalized_keypoints.tolist()
                # print(normalized_keypoints)
                # print(type(normalized_keypoints))
                insert_initial_pose_to_db(normalized_keypoints, class_name)


if __name__ == "__main__":
    run()