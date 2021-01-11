import gc
import csv
import time
import random
from datetime import datetime
from tensorflow.keras import regularizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter("ignore")

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

class ForceGarbageCollection(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

# Write headers
def write_header(filename):
    if not os.path.exists('multithreading_test'):
        os.mkdir('multithreading_test')
    f = open('multithreading_test/' + filename + '.csv', 'w')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'lstm_layer', 'n_hidden', 'seconds_to_finish', 'k-fold 1', 'k-fold 1 time', 'k-fold 2', 'k-fold 2 time', 'k-fold 3', 'k-fold 3 time', 'k-fold 4', 'k-fold 4 time', 'k-fold 5', 'k-fold 5 time', 'avg']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(filename, data):
    if not os.path.exists('multithreading_test'):
        os.mkdir('multithreading_test')
    f = open('multithreading_test/' + filename + '.csv', 'a')
    with f:
        fnames = ['exercise name', 'epoch', 'batch_size', 'dropout', 'lstm_layer', 'n_hidden', 'seconds_to_finish', 'k-fold 1', 'k-fold 1 time', 'k-fold 2', 'k-fold 2 time', 'k-fold 3', 'k-fold 3 time', 'k-fold 4', 'k-fold 4 time', 'k-fold 5', 'k-fold 5 time', 'avg']
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
        writer.writeheader()   
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
    n_splits = 5

    # Initialize K Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    k_fold_index = 1

    x = np.array(x)
    y = np.array(y)
    t_start = time.time()
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
        fit_start = time.time()
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.4, callbacks=[ForceGarbageCollection()])

        # Print model stats
        print(model.summary())

        # Find accuracy
        _, accuracy = model.evaluate(x_test, y_test)
        accuracy *= 100
        total += accuracy
        body[f'k-fold {k_fold_index}'] = "{:.2f}".format(accuracy)
        body[f'k-fold {k_fold_index} time'] = float(time.time() - fit_start)
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

    def run(thread_name, type_name, n_hidden, x, y):
        lstm_layers = [4]
        dropouts = [0.5]
        epochs = [250]
        batch_sizes = [100]

        for lstm_layer in lstm_layers:
            for dropout in dropouts:
                for epoch in epochs:
                    for batch_size in batch_sizes:
                        filename = f'{thread_name}_lstm_{type_name}_hidden_{n_hidden}_layers_{lstm_layer}_dropout_{dropout}_epoch_{epoch}_batch_size_{batch_size}'
                        date_string = datetime.now().isoformat().replace(':', '.')
                        log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/multithreading_test/training_logs/"
                        sys.stdout= open(os.path.join(log_dir, f'{filename}-{date_string}.txt'), 'w')
                        train(type_name, filename, n_hidden, lstm_layer, dropout, epoch, batch_size, x, y)

    type_name = "push-up"
    THREADS = []
    hidden = [44, 22, 11]
    
    x, y = get_dataset_by_type(type_name)

    print("Starting Multithreading Countdown...")
    test_start = time.time()
    for n_hidden in hidden:
        thread = Process(target=run, args=("multi", type_name, n_hidden, x, y,))
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
    body['exercise_name'] = "lstm pose evaluator"
    body['thread_type'] = "multithreaded"
    body['time_start'] = f"{float(test_start)}"
    body['time_end'] = f"{float(test_end)}"
    body['seconds_to_finish'] = f"{float(test_end - test_start)}"
    write_time("multithreaded_lstm_pose_time", body)
    
    print("\nStarting Singlethreaded Countdown...")
    test_start = time.time()
    for n_hidden in hidden:
        run("single", type_name, n_hidden, x, y)
    test_end = time.time()
    print("\n\n\n==========================================")
    print(f"Singlethreaded Start Time (seconds): {float(test_start)}")
    print(f"Singlethreaded End Time (seconds): {float(test_end)}")
    print(f"\nSinglethreaded Interval Time (seconds): {float(test_end - test_start)}")
    print("==========================================\n\n\n")

    body = {}
    body['exercise_name'] = "lstm pose evaluator"
    body['thread_type'] = "singlethreaded"
    body['time_start'] = f"{float(test_start)}"
    body['time_end'] = f"{float(test_end)}"
    body['seconds_to_finish'] = f"{float(test_end - test_start)}"
    write_time("singlethreaded_lstm_pose_time", body)
