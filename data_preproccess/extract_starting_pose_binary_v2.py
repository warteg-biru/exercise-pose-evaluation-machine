import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')
from keypoints_extractor import KeypointsExtractor
import os
import cv2
import urllib
from pymongo import MongoClient

def connect_to_mongo():
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        return db

    except Exception as e:
        print("Could not connect to MongoDB " , e)


def insert_starting_pose_binary_to_db(kp_array, exercise_name, is_starting_pose):
    db = connect_to_mongo()
    collection = db["starting_pose_binary_v2"]

    try:
        data = {
            "keypoints": kp_array,
            "exercise_name": exercise_name,
            "is_starting_pose": is_starting_pose,
        }
        # Insert into database collection
        rec_id1 = collection.insert_one(data) 
        print("Data inserted with record id",rec_id1)
    except Exception as e:
        print("Failed to insert data to database, errors: ", e)


def validate_keypoints(keypoints):
    if len(keypoints) != 14:
        print("Data invalid! Expected 14 keypoints, received " + str(len(keypoints)) + ".")
        return False
    for coordinates in keypoints:
        if len(coordinates) != 2:
            print("Data invalid! Expected 2 coordinates, received " + str(len(coordinates)) + ".")
            return False
    return True

def run():
    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()

    base_path = "/home/kevin/projects/initial-pose-data/images/v2/pos-negs"
    for exercise_name in os.listdir(base_path):
        exercise_dir = os.path.join(base_path, exercise_name)
        for posneg in os.listdir(exercise_dir):
            posneg_dir = os.path.join(exercise_dir, posneg)
            
            for file in os.listdir(posneg_dir):
                file_path = os.path.join(posneg_dir, file)
                image = cv2.imread(file_path)
                list_of_pose_and_id, _ = kp_extractor.get_keypoints_and_id_from_img(image)
                keypoints = list_of_pose_and_id[0]['Keypoints']
                if validate_keypoints(keypoints):
                    keypoints = keypoints.tolist()
                    is_starting_pose = True if posneg == "pos" else False
                    insert_starting_pose_binary_to_db(keypoints, exercise_name, is_starting_pose)

if __name__ == "__main__":
    run()
