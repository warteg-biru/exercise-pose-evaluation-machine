import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

import sys
# sys.path.append('/home/lab-mhs2/exercise_pose_evaluation_machine/')
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')
from db_entity import get_right_hand_up_dataset

def load_right_hand_up_pose_model():
    model_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up/right_hand_up.h5'
    model = load_model(model_path)
    return model


def get_dataset():
    x = []
    y = []

    # Get dataset
    true_dataset = get_right_hand_up_dataset(True)
    false_dataset = get_right_hand_up_dataset(False)

    # Use the same amount of data for true and false
    true_len = len(true_dataset)
    false_len = len(false_dataset)
    max_len = true_len if true_len > false_len else false_len
    true_dataset = true_dataset[:max_len]
    false_dataset = false_dataset[:max_len]

    # Loop in each folder
    for keypoints in true_dataset:
        x.append(np.array(keypoints).flatten())
        y.append(1)
    for keypoints in false_dataset:
        x.append(np.array(keypoints).flatten())
        y.append(0)

    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)

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

    print(x_test)
    
    model = load_right_hand_up_pose_model()
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
    plt.savefig("confusion_matrices/right_hand_up_conf_mat.png")

if __name__ == '__main__':
    main()

