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
    if not os.path.exists('k-fold-results'):
        os.mkdir('k-fold-results')
    f = open(f'k-fold-results/{filename}.csv', 'w')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write body
def write_body(filename, data):
    if not os.path.exists('k-fold-results'):
        os.mkdir('k-fold-results')
    f = open(f'k-fold-results/{filename}.csv', 'a')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10', 'avg']
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
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.3)

        # Find accuracy
        _, accuracy = model.evaluate(x_test, y_test)
        accuracy *= 100
        total += accuracy
        body['k-fold ' + str(k_fold_index)] = "{:.2f}".format(accuracy)
        print('Accuracy: %.2f' % (accuracy))
        k_fold_index += 1

    # Write iterations
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

    def run(epoch, batch_size, dropout, double, x, y):
        name = f'initial_pose_detector_{epoch}_epoch_{batch_size}_batch_size_{dropout}_dropout'
        if double:
            name += '_2x30'
        date_string = datetime.now().isoformat().replace(':', '.')
        print("Starting " + name)
        log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/k_fold_results/training_logs/"
        sys.stdout= open(os.path.join(log_dir, f'{name}-{date_string}.txt'), 'w')
        train(name, epoch, batch_size, dropout, double, x, y)
        print("Exiting " + name)

    THREADS = []
    epochs = [100, 150, 200, 250]
    batch_sizes = [10, 25, 50, 100]
    dropouts = [0.1, 0.2, 0.3]

    x, y = get_dataset()

    for epoch in epochs:
        for batch_size in batch_sizes:
            for dropout in dropouts:
                thread = Process(target=run, args=(epoch, batch_size, dropout, False, x, y,))
                thread.start()
                THREADS.append(thread)

                thread = Process(target=run, args=(epoch, batch_size, dropout, True, x, y,))
                thread.start()
                THREADS.append(thread)
            for t in THREADS:
                t.join()
            pop_all(THREADS)