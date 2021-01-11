import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')
import os
import cv2
from keypoints_extractor import KeypointsExtractor
from sklearn.preprocessing import MinMaxScaler
from db_entity import insert_starting_pose_binary_to_db

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

    base_path = "/home/kevin/projects/initial-pose-data/images/pos-negs"
    for exercise_name in os.listdir(base_path):
        exercise_dir = os.path.join(base_path, exercise_name)
        for posneg in os.listdir(exercise_dir):
            posneg_dir = os.path.join(exercise_dir, posneg)
            
            for file in os.listdir(posneg_dir):
                file_path = os.path.join(posneg_dir, file)
                image = cv2.imread(file_path)
                try:
                    list_of_pose_and_id, _ = kp_extractor.get_keypoints_and_id_from_img(image)
                except Exception as e:
                    # cv2.imshow("image yg error", image)
                    # cv2.waitKey(0)
                    print(e)
                    continue
                keypoints = list_of_pose_and_id[0]['Keypoints']
                if validate_keypoints(keypoints):
                    keypoints = keypoints.tolist()
                    is_starting_pose = True if posneg == "pos" else False
                    insert_starting_pose_binary_to_db(keypoints, exercise_name, is_starting_pose)
                else:
                    cv2.imshow("image yg ga ke validate", image)
                    cv2.waitKey(0)

if __name__ == "__main__":
    run()
