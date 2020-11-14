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
from list_manipulator import pop_all

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from db_entity import get_starting_pose_binary_from_db
from tensorflow import keras as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Permute, Reshape, Dense, Dropout
from tensorflow.keras.optimizers import SGD

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

def train(exercise_name, epochs, batch_size, double, dataset):
    # Get keypoint
    x = []
    y = []
    for data in dataset:
        keypoints = np.array(data["keypoints"]).flatten()
        x.append(keypoints)

        is_starting_pose = data["is_starting_pose"]
        label = 1 if is_starting_pose else 0
        y.append(label)

    # Initialize paths
    base_path = "/home/kevin/projects/initial-pose-data/train_data"
    date_string = datetime.now().isoformat()
    filename = f'{exercise_name}_{epochs}_epochs_{batch_size}_batch_size_2x30 binary pose k-fold results {date_string}' if double else f'{exercise_name}_{epochs}_epochs_{batch_size}_batch_size binary pose k-fold results {date_string}'

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
        num_output = 1

        # Decaying learning rate
        learning_rate = 0.01
        lr_schedule = PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10,
            end_learning_rate= 0.00001
        )
        optimizer = SGD(learning_rate = lr_schedule)

        model = Sequential()
        model.add(Dense(60, input_shape=(num_features,)))
        model.add(Dense(30, activation='relu'))
        if double:
            model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_output, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # Train model
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.3)

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
        # "plank",
        "squat"
    ]

    def run(type_name, epoch, batch_size, double, dataset):
        name = f'{type_name}_{epoch}_epoch_{batch_size}_batch_size'
        if double:
            name += '_2x30'
        date_string = datetime.now().isoformat()
        print("Starting " + name)
        log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/k_fold_results/training_logs/"
        sys.stdout= open(os.path.join(log_dir, f'{name}-binary-pose-{date_string}.txt'), 'w')
        train(type_name, epoch, batch_size, double, dataset)
        print("Exiting " + name)

    THREADS = []
    epochs = [100, 150, 200, 250]
    batch_sizes = [25, 50, 100]

    for type_name in exercise_names:
        # Get dataset
        data = get_starting_pose_binary_from_db(type_name)
        for epoch in epochs:
            for batch_size in batch_sizes:
                thread = Process(target=run, args=(type_name, epoch, batch_size, False, data))
                thread.start()
                thread = Process(target=run, args=(type_name, epoch, batch_size, True, data))
                thread.start()
                THREADS.append(thread)
            for t in THREADS:
                t.join()
            pop_all(THREADS)