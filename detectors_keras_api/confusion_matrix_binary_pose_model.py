import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import os

import sys
# sys.path.append('/home/lab-mhs2/exercise_pose_evaluation_machine/')
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

from db_entity import get_starting_pose_binary_from_db


def load_saved_model(exercise_name):
    model_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model/' + str(exercise_name) + '/' + str(exercise_name) + '_pose_model.h5'
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model

def get_dataset(exercise_name):
    dataset = get_starting_pose_binary_from_db(exercise_name)

    x = []
    y = []
    for data in dataset:
        keypoints = np.array(data["keypoints"]).flatten()
        x.append(keypoints)

        is_starting_pose = data["is_starting_pose"]
        label = 1 if is_starting_pose else 0
        y.append(label)

    return x, y

def run(exercise_name):
    x, y = get_dataset(exercise_name)

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    x_test = []
    y_test = []
    x = np.array(x)
    y = np.array(y)
    for train_index, test_index in skf.split(x, y):
        x_test.append(x[test_index])
        y_test.append(y[test_index])

    model = load_saved_model(exercise_name)
    y_pred = []
    for x in x_test:
        prediction = model.predict(x)
        for p in prediction:
            val = 1 if p[0] > 0.5 else 0
            y_pred.append(val)

    # Flatten
    y_test = [y for inner_list in y_test for y in inner_list]

    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, cmap="Blues")
    plt.savefig("confusion_matrices/" + exercise_name + "_binary_pos_conf_mat.png")
    plt.clf()


def main():
    exercise_names = [
        "sit-up",
        "push-up",
        "squat"
    ]
    for ex in exercise_names:
        run(ex)
    

if __name__ == '__main__':
    main()

