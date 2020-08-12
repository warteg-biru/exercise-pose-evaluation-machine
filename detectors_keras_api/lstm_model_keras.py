import os
import time
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import collections
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from db_entity import get_dataset
from keypoints_extractor import pop_all

from sklearn.model_selection import train_test_split

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Permute, Reshape

CLASS_TYPE = [
    "push-up2",
]

for type_name in CLASS_TYPE:
    # Get original dataset
    x, y = get_dataset(type_name)

    # Fill original class type with the label 1
    y = [1 for label in y]

    # Get negative dataset
    neg_x, neg_y = get_dataset("plank2")
    # Fill original class type with the label 1
    neg_y = [0 for label in neg_y]
    x.extend(neg_x)
    y.extend(neg_y)

    # Get negative dataset
    neg_x, neg_y = get_dataset("situp2")
    neg_y = [2 for label in neg_y]
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

    # Split to training and test dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    n_steps = 24
    n_input = len(x_train[0][0])
    n_hidden = 22
    n_classes = 3

    # pair of lstm cell initialization through loop
    lstm_cells = [LSTMCell(
        n_hidden,
        activation='relu',
        use_bias=True,
        unit_forget_bias = 1.0
    ) for _ in range(2)]
    stacked_lstm = StackedRNNCells(lstm_cells)
    lstm_layer = RNN(stacked_lstm)  


    model = Sequential()
    model.add(lstm_layer)


    print(model.summary())