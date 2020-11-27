import cv2
import csv
import time
import traceback

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.simplefilter("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras import regularizers
from datetime import datetime

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

import collections
import random
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
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
    if not os.path.exists('multithreading_test'):
        os.mkdir('multithreading_test')
    f = open(f'multithreading_test/{filename}.csv', 'w')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'seconds_to_finish', 'k-fold 1', 'k-fold 1 time', 'k-fold 2', 'k-fold 2 time', 'k-fold 3', 'k-fold 3 time', 'k-fold 4', 'k-fold 4 time', 'k-fold 5', 'k-fold 5 time', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write body
def write_body(filename, data):
    if not os.path.exists('multithreading_test'):
        os.mkdir('multithreading_test')
    f = open(f'multithreading_test/{filename}.csv', 'a')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'seconds_to_finish', 'k-fold 1', 'k-fold 1 time', 'k-fold 2', 'k-fold 2 time', 'k-fold 3', 'k-fold 3 time', 'k-fold 4', 'k-fold 4 time', 'k-fold 5', 'k-fold 5 time', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

# Write time
def write_time(filename, data):
    if not os.path.exists('multithreading_test'):
        os.mkdir('multithreading_test')
    f = open('multithreading_test/' + filename + '.csv', 'a')
    with f:
        fnames = ['exercise_name', 'thread_type', 'time_start', 'time_end', 'seconds_to_finish']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

def train(filename, epoch, batch_size, dropout, double, x, y):
    # Initialize paths
    date_string = datetime.now().isoformat().replace(':', '.')
    filename = f'{filename} k-fold results {date_string}'

    # Create file and write CSV header
    write_header(filename)
    body = {}
    body['epoch'] = epoch
    body['batch_size'] = batch_size
    body['dropout'] = dropout

    # Initialize total accuracy variable and number of K-Fold splits
    total = 0
    n_splits = 5

    # Initialize K Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    k_fold_index = 1

    x = np.array(x)
    y = np.array(y)
    t_start = time.time()
    for train_index, test_index in skf.split(x, y):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]        
        
        # Define number of features, labels, and hidden
        num_features = 28 # 14 pairs of (x, y) keypoints
        num_labels = 5
        
        '''
        build_model

        # Builds an ANN model for keypoint predictions
        @params {list of labels} image prediction labels to be tested
        @params {integer} number of features
        @params {integer} number of labels as output
        @params {integer} number of hidden layers
        '''
        model = Sequential()
        model.add(Dense(60, input_shape=(num_features,)))
        model.add(Dense(30, activation='relu'))
        if double:
            model.add(Dense(30, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(num_labels, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        
        # Train model
        fit_start = time.time()
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.3)

        # Find accuracy
        _, accuracy = model.evaluate(x_test, y_test)
        accuracy *= 100
        total += accuracy
        body[f'k-fold {k_fold_index}'] = "{:.2f}".format(accuracy)
        body[f'k-fold {k_fold_index} time'] = float(time.time() - fit_start)
        print('Accuracy: %.2f' % (accuracy))
        k_fold_index += 1

    # Write iterations
    body['seconds_to_finish'] = float(time.time() - t_start)
    body['exercise name'] = "initial pose detector"
    body['avg'] = "{:.2f}".format(total/n_splits)
    write_body(filename, body)

def get_dataset():    
    # Get data from mongodb
    exercise_name_labels = { "sit-up": 0, "plank": 1, "squat": 2, "push-up": 3, "stand": 4 }
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

if __name__ == '__main__':
    from multiprocessing import Process

    def run(thread_type, epoch, batch_size, dropout, double, x, y):
        name = f'{thread_type}_initial_pose_detector_{epoch}_epoch_{batch_size}_batch_size_{dropout}_dropout'
        if double:
            name += '_2x30'
        date_string = datetime.now().isoformat().replace(':', '.')
        log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/multithreading_test/training_logs/"
        sys.stdout= open(os.path.join(log_dir, f'{name}-{date_string}.txt'), 'w')
        train(name, epoch, batch_size, dropout, double, x, y)

    THREADS = []
    epochs = [250]
    batch_sizes = [10, 25, 50]
    dropouts = [0.2]

    x, y = get_dataset()

    print("Starting Multithreading Countdown...")
    test_start = time.time()
    for epoch in epochs:
        for dropout in dropouts:
            for batch_size in batch_sizes:
                thread = Process(target=run, args=("multi", epoch, batch_size, dropout, False, x, y,))
                thread.start()
                THREADS.append(thread)

                thread = Process(target=run, args=("multi", epoch, batch_size, dropout, True, x, y,))
                thread.start()
                THREADS.append(thread)
            for t in THREADS:
                t.join()
            pop_all(THREADS)
    test_end = time.time()
    print("\n\n\n==========================================")
    print(f"Multithreading Start Time (seconds): {float(test_start)}")
    print(f"Multithreading End Time (seconds): {float(test_end)}")
    print(f"\nMultithreading Interval Time (seconds): {float(test_end - test_start)}")
    print("==========================================\n\n\n")

    body = {}
    body['exercise_name'] = "initial pose detector"
    body['thread_type'] = "multithreaded"
    body['time_start'] = f"{float(test_start)}"
    body['time_end'] = f"{float(test_end)}"
    body['seconds_to_finish'] = f"{float(test_end - test_start)}"
    write_time("multithreaded_init_pose_time", body)
    
    print("\nStarting Singlethreaded Countdown...")
    test_start = time.time()
    for epoch in epochs:
        for dropout in dropouts:
            for batch_size in batch_sizes:
                run("single", epoch, batch_size, dropout, False, x, y)
                run("single", epoch, batch_size, dropout, True, x, y)
    test_end = time.time()
    print("\n\n\n==========================================")
    print(f"Singlethreaded Start Time (seconds): {float(test_start)}")
    print(f"Singlethreaded End Time (seconds): {float(test_end)}")
    print(f"\nSinglethreaded Interval Time (seconds): {float(test_end - test_start)}")
    print("==========================================\n\n\n")

    body = {}
    body['exercise_name'] = "initial pose detector"
    body['thread_type'] = "singlethreaded"
    body['time_start'] = f"{float(test_start)}"
    body['time_end'] = f"{float(test_end)}"
    body['seconds_to_finish'] = f"{float(test_end - test_start)}"
    write_time("singlethreaded_init_pose_time", body)