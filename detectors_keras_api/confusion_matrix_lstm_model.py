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

from db_entity import get_dataset


def load_saved_model(type_name):
    model_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/lstm_model/keras/' + type_name + '/' + type_name + '_lstm_model.h5'
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model


def get_dataset_by_type(type_name):
    # Get original dataset
    x, y = get_dataset(type_name)
    # Fill original class type with the label 1
    y = [1 for label in y]

    # Get negative dataset
    neg_x, neg_y = get_dataset("not-" + type_name)
    
    # Fill original class type with the label 1
    neg_y = [0 for label in neg_y]
    x.extend(neg_x)
    y.extend(neg_y)

    # Flatten X coodinates and filter
    x = np.array(x)
    _x = []
    _y = []
    for idx, data in enumerate(x):
        data = [np.reshape(np.array(frames), (28)).tolist() for frames in data]
        _x.append(data)
        _y.append(y[idx])
    x = _x
    y = _y
    return x, y


def run(exercise_name):
    x, y = get_dataset_by_type(exercise_name)

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
    sns.heatmap(conf_mat/np.sum(conf_mat), annot=True, fmt='.2%', cmap="Blues")
    plt.savefig("confusion_matrices/" + exercise_name + "_lstm_model_conf_mat.png")
    plt.clf()


def main():
    exercise_names = [
        "push-up",
        "sit-up",
        "plank",
        "squat"
    ]
    for ex in exercise_names:
        run(ex)
    

if __name__ == '__main__':
    main()


