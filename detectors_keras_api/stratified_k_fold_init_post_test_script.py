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
    if not os.path.exists('k-fold-results'):
        os.mkdir('k-fold-results')
    f = open('k-fold-results/' + filename + '.csv', 'w')
    with f:
        fnames = ['exercise name', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write body
def write_body(filename, data):
    if not os.path.exists('k-fold-results'):
        os.mkdir('k-fold-results')
    f = open('k-fold-results/' + filename + f'.csv', 'a')
    with f:
        fnames = ['exercise name', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

def train():
    # Initialize paths
    base_path = "/home/kevin/projects/initial-pose-data/train_data"
    save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5"
    date_string = datetime.now().isoformat()
    filename = f'initial pose model k-fold results {date_string}'

    # Get dataset folders
    dirs = os.listdir(base_path)

    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()

    x = []
    y = []
    
    # get data from mongodb
    exercise_name_labels = { "sit-up": 0, "plank": 1, "squat": 2, "push-up": 3, "stand": 4 }
    x = []
    y = []
    dataset = get_initial_pose_dataset()
    
    for exercise_name, keypoints in dataset.items():
        keypoints = [np.array(kp).flatten() for kp in keypoints]
        for kp in keypoints:
            # print(kp.shape)
            # import time
            # time.sleep(20)
            x.append(kp)
            y.append(exercise_name_labels[exercise_name])

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
    n_splits = 5

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
        model.fit(x_train, y_train, epochs=200, batch_size=10)

        # Find accuracy
        _, accuracy = model.evaluate(x_train, y_train)
        accuracy *= 100
        total += accuracy
        body['k-fold ' + str(k_fold_index)] = "{:.2f}".format(accuracy)
        print('Accuracy: %.2f' % (accuracy))
        k_fold_index += 1

    # Write iterations
    body['exercise name'] = "initial pose detector"
    body['avg'] = "{:.2f}".format(total/n_splits)
    write_body(filename, body)

if __name__ == '__main__':
    log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/k-fold-results/training_logs/"
    date_string = datetime.now().isoformat()
    sys.stdout= open(os.path.join(log_dir, f'{"intital starting pose"}-{date_string}.txt'), 'w')
    train()
    sys.stdout.close()