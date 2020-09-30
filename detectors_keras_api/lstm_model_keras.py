import os
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

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Permute, Reshape, Dense, Dropout
from datetime import datetime

def load_saved_model(model_path):
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model
    raise Exception('model does not exist')


def save_model_summary(summary):
    training_logs_dir = "/home/"
    f = open()



class ExerciseEvaluator:
    def __init__(self, type_name):
        MODEL_PATH = '/home/kevin/projects/exercise_pose_evaluation_machine/models/lstm_model/keras/' + type_name + '/' + type_name + '_lstm_model.h5'
        self.model = load_saved_model(MODEL_PATH)
        self.type_name = type_name

    def predict(self, data):
        try:
            if self.type_name is "sit-up":
                assert data.shape == (1, 48)
            else:
                assert data.shape == (1, 24)
            return "1" if self.model.predict(np.array([data])) > 0.5 else "0"
        except BaseException as e:
            traceback.print_exc()

def train(type_name, n_hidden):
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
        data = [np.reshape(np.array(frames), (28)).tolist() for frames in data]
        _x.append(data)
        _y.append(y[idx])
    x = _x
    y = _y

    # Split to training and test dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Define training parameters
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
    model.fit(x_train, y_train, epochs=25, batch_size=10, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.4)

    # Print model stats
    print(model.summary())
    print(model.get_config())

    # Find accuracy
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    # Save model
    model.save(save_path)
    print("Saved model!")

    # Generate predictions
    print("See prediction result")
    random_int = random.randint(0, len(x_test))
    data = x_test[random_int]
    prediction = "1" if model.predict(np.array([data])) > 0.5 else "0"
    print("predictions result:", prediction)
    print("expected result: ", y_test[random_int])

    # UNTUK SELANJUTNYA, DIBUAT TRY EXCEPT UNTUK SETIAP BLOCK BERBEDA
    # SEPERTI SAAT PREDICT ATAU SAAT OLAH DATA ATAUPUN SAAT CEK AKURASI
    # AGAR GAMPANG PINPOINT MASALAH.

# Initiate function
if __name__ == '__main__':
    CLASS_TYPE = [
        "push-up",
        "sit-up",
        "plank"
    ]

    HIDDEN_NUM = [44,22,11]

    for type_name in CLASS_TYPE:
        for hidden in HIDDEN_NUM:
            log_dir = "/home/kevin/projects/exercise_pose_evaluation_machine/models/training_logs/"
            date_string = datetime.now().isoformat()
            sys.stdout= open(os.path.join(log_dir, f'{type_name}-{date_string}.txt'), 'w')
            train(type_name, hidden)
            sys.stdout.close()