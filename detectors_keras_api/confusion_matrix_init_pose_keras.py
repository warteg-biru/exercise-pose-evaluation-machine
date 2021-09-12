import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

import sys
# sys.path.append('/home/lab-mhs2/exercise_pose_evaluation_machine/')
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')
from db_entity import get_initial_pose_dataset

def load_initial_pose_model():
    model_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5"
    model = load_model(model_path)
    return model


def get_dataset():
    # Get data from mongodb
    # TODO: Exclude stand from dataset
    exercise_name_labels = { "plank": 0, "push-up": 1, "sit-up": 2, "squat": 3, "stand": 4 }
    x = []
    y = []
    dataset = get_initial_pose_dataset()
    
    for exercise_name, keypoints in dataset.items():
        keypoints = [np.array(kp).flatten() for kp in keypoints]
        for kp in keypoints:
            x.append(kp)
            y.append(exercise_name_labels[exercise_name])
    y = np.array(y)
    return x, y


def main():
    x, y = get_dataset()

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    x_test = []
    y_test = []
    x = np.array(x)
    y = np.array(y)
    for train_index, test_index in skf.split(x, y):
        x_test.append(x[test_index])
        y_test.append(y[test_index])

    
    model = load_initial_pose_model()
    y_pred = []
    for x in x_test:
        prediction = model.predict(x)
        for p in prediction:
            y_pred.append(np.argmax(p))

    # Flatten
    y_test = [y for inner_list in y_test for y in inner_list]

    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, cmap="Blues")
    plt.savefig("init_pos_conf_mat.png")

if __name__ == '__main__':
    main()
