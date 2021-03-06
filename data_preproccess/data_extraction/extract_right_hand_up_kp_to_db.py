import sys
import os
import cv2
import time
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

ROOT_DIR = "/home/kevin/projects/exercise_pose_evaluation_machine"
sys.path.append(ROOT_DIR)

from keypoints_extractor import KeypointsExtractor
from db_entity import insert_right_hand_up_pose_to_db

from logs.csv_logger import CSVLogger
from list_manipulator import pop_all, chunks

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# Validate Upper body keypoints. Should have 8 elements
def validate_keypoints(keypoints):
    if len(keypoints) != 13:
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
            ROOT_DIR, f"logs/kp_extraction_logs/upper_body_{date_string}.csv"
        ),
        headers=["image_res", "time"]
    )
    return logger

def run(base_path, label, label_dir, kp_extractor, min_max_scaler):
    for file_name in label_dir:
        file_path = f'{base_path}/{label}/{file_name}'

        image = cv2.imread(file_path)
        # Get image dimensions for logging
        height, width, channels = image.shape
        
        try:
            # record the time to extract keypoints
            keypoints = kp_extractor.get_upper_body_keypoint(file_path)
        except:
            continue
        
        if validate_keypoints(keypoints):
            normalized_keypoints = [x.tolist() for x in keypoints]
            insert_right_hand_up_pose_to_db(normalized_keypoints, label)


if __name__ == "__main__":
    from multiprocessing import Process

    # Initialize paths
    # TODO: Setup the train data in 2 folders true and false folders
    base_path = "/home/kevin/projects/right_hand_up_pose_data/training_data"

    # Get dataset folders
    dirs = os.listdir(base_path)

    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()
    min_max_scaler = MinMaxScaler(feature_range=(0,1))

    # Label should be either "true" / "false" (Which are the folder names)
    for label in dirs:
        label_dir = os.listdir(base_path+'/'+label)
        run(base_path, label, label_dir, kp_extractor, min_max_scaler)

        # label_dir_split = chunks(label_dir, int(len(label_dir) / 3))
        # THREADS = []

        # for label_dir in label_dir_split:
        #     thread = Process(target=run, args=(base_path, label, label_dir,))
        #     thread.start()
        #     THREADS.append(thread)
        # for t in THREADS:
        #     t.join()
        # pop_all(THREADS)
