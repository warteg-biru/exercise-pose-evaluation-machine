import csv
import time
import random
from datetime import datetime
from tensorflow.keras import regularizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

import time
import random
import numpy as np
import collections
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Permute, Reshape, Dense, Dropout

# Write headers
def write_header(filename):
    if not os.path.exists('k_fold_results'):
        os.mkdir('k_fold_results')
    f = open('k_fold_results/' + filename + '.csv', 'w')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'lstm_layer', 'n_hidden', 'seconds_to_finish', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(filename, data):
    if not os.path.exists('k_fold_results'):
        os.mkdir('k_fold_results')
    f = open('k_fold_results/' + filename + '.csv', 'a')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'lstm_layer', 'n_hidden', 'seconds_to_finish', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

def train(type_name, filename, n_hidden, lstm_layer, dropout, epoch, batch_size, x, y):
    # Make filename
    date_string = datetime.now().isoformat().replace(':', '.')
    filename = f'{filename} k-fold results {date_string}'

    # Create file and write CSV header
    write_header(filename)
    body = {}
    body['n_hidden'] = n_hidden
    body['lstm_layer'] = lstm_layer
    body['dropout'] = dropout
    body['epoch'] = epoch
    body['batch_size'] = batch_size

    # Initialize total accuracy variable and number of K-Fold splits
    total = 0
    n_splits = 10
    t_start = time.time()

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
        n_output = 1

        # Make LSTM Layer
        # Pair of lstm cell initialization through loop
        lstm_cells = [LSTMCell(
            n_hidden,
            activation='relu',
            use_bias=True,
            unit_forget_bias = 1.0
        ) for _ in range(lstm_layer)]
        stacked_lstm = StackedRNNCells(lstm_cells)

        # Decaying learning rate
        learning_rate = 1e-2
        lr_schedule = PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10,
            end_learning_rate= 0.00001
        )
        optimizer = Adam(learning_rate = lr_schedule)

        # Initiate model
        model = Sequential()
        model.add(RNN(stacked_lstm))
        model.add(Dropout(dropout))
        model.add(Dense(n_output, 
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l1(0.01)))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Train model
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.4)

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
    body['seconds_to_finish'] = float(time.time() - t_start)
    body['exercise name'] = type_name
    body['avg'] = "{:.2f}".format(total/n_splits)
    write_body(filename, body)

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

if __name__ == '__main__':
    from multiprocessing import Process

    def run(type_name, x, y):
        hidden = [44, 22, 11]
        lstm_layers = [2,3,4]
        dropouts = [0.3, 0.4, 0.5, 0.6]
        epochs = [200, 250, 300]
        batch_sizes = [100, 150, 200]

        for n_hidden in hidden:
            for lstm_layer in lstm_layers:
                for dropout in dropouts:
                    for epoch in epochs:
                        for batch_size in batch_sizes:
                            filename = f'lstm_{type_name}_hidden_{n_hidden}_layers_{lstm_layer}_dropout_{dropout}_epoch_{epoch}_batch_size_{batch_size}'
                            date_string = datetime.now().isoformat().replace(':', '.')
                            print("Starting " + filename)
                            log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/k_fold_results/training_logs/"
                            sys.stdout= open(os.path.join(log_dir, f'{filename}-{date_string}.txt'), 'w')
                            train(type_name, filename, n_hidden, lstm_layer, dropout, epoch, batch_size, x, y)
                            print("Exiting " + filename)

    exercise_names = [
        "push-up",
        "sit-up",
        "plank",
        "squat"
    ]

    THREADS = []

    for type_name in exercise_names:
        x, y = get_dataset_by_type(type_name)
        thread = Process(target=run, args=(type_name, x, y,))
        thread.start()
        THREADS.append(thread)
    for t in THREADS:
        t.join()
    pop_all(THREADS)