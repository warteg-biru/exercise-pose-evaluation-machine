import os
import csv
import time
import cv2
import traceback

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras import regularizers
from datetime import datetime

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

import collections
import random
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from db_entity import get_right_hand_up_dataset
from keypoints_extractor import pop_all

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from db_entity import get_initial_pose_dataset
from tensorflow import keras as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Permute, Reshape, Dense, Dropout
from tensorflow.keras.optimizers import SGD

from keypoints_extractor import KeypointsExtractor
from right_hand_detector_keras import get_dataset, create_model
# Write headers
def write_header(filename):
    if not os.path.exists('k_fold_results'):
        os.mkdir('k_fold_results')
    f = open('k_fold_results/' + filename + '.csv', 'w')
    with f:
        fnames = ['exercise name', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write body
def write_body(filename, data):
    if not os.path.exists('k_fold_results'):
        os.mkdir('k_fold_results')
    f = open('k_fold_results/' + filename + f'.csv', 'a')
    with f:
        fnames = ['exercise name', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

def train():
    # Initialize paths
    save_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up/right_hand_up.h5'
    date_string = datetime.now().isoformat()
    filename = f'right hand up model k-fold results {date_string}'
    
    x = []
    y = []
    dataset = get_right_hand_up_dataset()
    for label, keypoints in dataset.items():
        keypoints = [np.array(kp).flatten() for kp in keypoints]
        for kp in keypoints:
            x.append(kp)
            if label == 'true':
                y.append(1)
            else:
                y.append(0)


    # Create file and write CSV header
    write_header(filename)
    body = {}

    # Initialize total accuracy variable and number of K-Fold splits
    total = 0
    n_splits = 10

    # Initialize K Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    k_fold_index = 1

    x = np.array(x)
    y = np.array(y)
    for train_index, test_index in skf.split(x, y):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]        
        
        model = create_model()
        
        model.fit(x_train, y_train, epochs=25, batch_size=10, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.3)
        _, accuracy = model.evaluate(x_train, y_train)

        # Find accuracy
        _, accuracy = model.evaluate(x_test, y_test)
        accuracy *= 100
        total += accuracy
        body['k-fold ' + str(k_fold_index)] = "{:.2f}".format(accuracy)
        print('Accuracy: %.2f' % (accuracy))
        k_fold_index += 1

        model.summary()

    # Write iterations
    body['exercise name'] = "right hand up"
    body['avg'] = "{:.2f}".format(total/n_splits)
    write_body(filename, body)

if __name__ == '__main__':
    log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/k_fold_results/training_logs/"
    date_string = datetime.now().isoformat()
    sys.stdout= open(os.path.join(log_dir, f'{"right hand up"}-{date_string}.txt'), 'w')
    train()
    sys.stdout.close()