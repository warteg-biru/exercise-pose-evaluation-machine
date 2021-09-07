import numpy as np
from tensorflow.keras.models import load_model

import sys
# sys.path.append('/home/lab-mhs2/exercise_pose_evaluation_machine/')
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')
from db_entity import get_initial_pose_dataset

def load_initial_pose_model():
    model_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5"
    model = load_model(model_path)
    return model

def load_data_set():
    dataset = get_initial_pose_dataset()
    x = []
    y = []
    for exercise_name in dataset:
        keypoints_list = dataset[exercise_name]
        for kp in keypoints_list:
            x.append(kp)
            y.append(exercise_name)
    return x, y

def main():
    x, y = load_data_set()

    #TODO: Reshape x into to fit model.predict

    #model = load_initial_pose_model()

    #y_pred = model.predict(np.array([x[0]]))

    #print(y_pred)


if __name__ == '__main__':
    main()
