import os
import csv
import time
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
from tensorflow import keras as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Permute, Reshape, Dense, Dropout

# Write headers
def write_header(filename):
    if not os.path.exists('k-fold-results'):
        os.mkdir('k-fold-results')
    f = open('k-fold-results/' + filename + ' k-fold results.csv', 'w')
    with f:
        fnames = ['exercise name', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(filename, data):
    if not os.path.exists('k-fold-results'):
        os.mkdir('k-fold-results')
    f = open('k-fold-results/' + filename + ' k-fold results.csv', 'a')
    with f:
        fnames = ['exercise name', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

if __name__ == '__main__':
    CLASS_TYPE = [
        "push-up",
        # "sit-up",
        "plank"
    ]

    for type_name in CLASS_TYPE:
        # Initialize save path
        save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/lstm_model/keras/" + type_name + "/" + type_name + "_lstm_model.h5"
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
            if len(data) == 24:
                data = [np.reshape(np.array(frames), (28)).tolist() for frames in data]
                _x.append(data)
                _y.append(y[idx])
        x = _x
        y = _y

        # Create file and write CSV header
        write_header(type_name)
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
            # Initialize training sets
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]

            # Define training parameters
            n_input = len(x_train[0][0])
            n_hidden = 22
            n_classes = 1

            # Make LSTM Layer
            # Pair of lstm cell initialization through loop
            lstm_cells = [LSTMCell(
                n_hidden,
                activation='relu',
                use_bias=True,
                unit_forget_bias = 1.0
            ) for _ in range(2)]
            stacked_lstm = StackedRNNCells(lstm_cells)
            lstm_layer = RNN(stacked_lstm)

            # Initiate model
            model = Sequential()
            model.add(lstm_layer)
            model.add(Dropout(0.5))
            model.add(Dense(n_classes, 
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Train model
            model.fit(x_train, y_train, epochs=1, batch_size=10, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.4)

            # Print model stats
            print(model.summary())

            # Find accuracy
            _, accuracy = model.evaluate(x_test, y_test)
            accuracy *= 100
            total += accuracy
            body['k-fold ' + str(k_fold_index)] = "{:.2f}".format(accuracy)
            print('Accuracy: %.2f' % (accuracy))
            k_fold_index += 1

            # UNTUK SELANJUTNYA, DIBUAT TRY EXCEPT UNTUK SETIAP BLOCK BERBEDA
            # SEPERTI SAAT PREDICT ATAU SAAT OLAH DATA ATAUPUN SAAT CEK AKURASI
            # AGAR GAMPANG PINPOINT MASALAH.

        # Write iterations
        body['exercise name'] = type_name
        body['avg'] = "{:.2f}".format(total/n_splits)
        write_body(type_name, body)