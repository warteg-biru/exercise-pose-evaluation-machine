import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import csv
import cv2
import time
import traceback

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from datetime import datetime
from tensorflow.keras import regularizers

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

import collections
import random
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from db_entity import get_dataset
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

def train(exercise_name, dataset):
    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()

    # Get keypoint
    x = []
    y = []
    for keypoint_name, keypoints in dataset.items():
        if keypoint_name == exercise_name:
            keypoints = [np.array(kp).flatten() for kp in keypoints]
            for kp in keypoints:
                x.append(kp)
                y.append(0)

    total_pos = len(y)
    neg_per_class = int(total_pos / 4)

    # Data label from mongodb
    for keypoint_name, keypoints in dataset.items():
        if keypoint_name != exercise_name:
            keypoints = keypoints if len(keypoints) < neg_per_class else keypoints[:neg_per_class]
            keypoints = [np.array(kp).flatten() for kp in keypoints]
            for kp in keypoints:
                x.append(kp)
                y.append(1)

    # Initialize paths
    base_path = "/home/kevin/projects/initial-pose-data/train_data"
    date_string = datetime.now().isoformat()
    filename = f'{exercise_name} binary pose k-fold results {date_string}'

    # Get dataset folders
    dirs = os.listdir(base_path)

    # One hot encoder
    y = np.array(y)
    # y = y.reshape(-1, 1)
    # one_hot = OneHotEncoder(sparse=False)
    # y = one_hot.fit_transform(y)

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
        
        # Define number of features, labels, and hidden
        num_features = 28 # 14 pairs of (x, y) keypoints
        num_hidden = 8
        num_labels = 5
        
        '''
        build_model

        # Builds an ANN model for keypoint predictions
        @params {list of labels} image prediction labels to be tested
        @params {integer} number of features
        @params {integer} number of labels as output
        @params {integer} number of hidden layers
        '''

        # Decaying learning rate
        learning_rate = 1e-2
        lr_schedule = PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10,
            end_learning_rate= 0.00001
        )
        optimizer = SGD(learning_rate = lr_schedule)

        model = Sequential()
        model.add(Dropout(0.2, input_shape=(num_features,)))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(num_hidden, activation='relu'))
        model.add(Dense(num_labels, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        
        # Train model
        model.fit(x_train, y_train, epochs=100, batch_size=10, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.3)

        # Find accuracy
        _, accuracy = model.evaluate(x_test, y_test)
        accuracy *= 100
        total += accuracy
        body['k-fold ' + str(k_fold_index)] = "{:.2f}".format(accuracy)
        print('Accuracy: %.2f' % (accuracy))
        k_fold_index += 1

    # Write iterations
    body['exercise name'] = exercise_name
    body['avg'] = "{:.2f}".format(total/n_splits)
    write_body(filename, body)

if __name__ == '__main__':
    from multiprocessing import Process

    # Exercise Labels
    exercise_names = [
        "sit-up",
        "push-up",
        "plank",
        "squat"
    ]
    # Get dataset
    data = get_initial_pose_dataset()

    def run(type_name, dataset):
        name = f'{type_name}'
        date_string = datetime.now().isoformat()
        print("Starting " + type_name)
        log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/k_fold_results/training_logs/"
        sys.stdout= open(os.path.join(log_dir, f'{type_name}-binary-pose-{date_string}.txt'), 'w')
        train(type_name, dataset)
        print("Exiting " + type_name)

    THREADS = []

    for type_name in exercise_names:
        thread = Process(target=run, args=(type_name,data))
        thread.start()
        THREADS.append(thread)
    for t in THREADS:
        t.join()
    pop_all(THREADS)