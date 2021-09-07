import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Dense, Dropout, InputLayer, LSTM

import sys
# sys.path.append('/home/lab-mhs2/exercise_pose_evaluation_machine/')
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')
from db_entity import get_dataset

def print_model(filename, n_hidden, lstm_layer, dropout, type_name):
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
    model.add(InputLayer(input_shape=x_train[0].shape))
    # model.add(RNN(stacked_lstm))
    model.add(LSTM(
        n_hidden,
        activation='relu',
        use_bias=True,
        unit_forget_bias = 1.0, 
        input_shape=x_train[0].shape, 
        return_sequences=True))
    for _ in range(lstm_layer-2):
        model.add(LSTM(
            n_hidden,
            activation='relu',
            use_bias=True,
            unit_forget_bias = 1.0, 
            return_sequences=True))
    model.add(LSTM(
        n_hidden,
        activation='relu',
        use_bias=True,
        unit_forget_bias = 1.0))
    model.add(Dropout(dropout))
    model.add(Dense(n_output, 
        activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.01)))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Train model
    model.fit(x_train, y_train, epochs=1, batch_size=1000, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.4)

    # Print model
    with open(f'{filename}.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file=f'{filename}.png', show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
    print_model('lstm_model_smallest_parameters', 11, 2, 0.3, "push-up")
    print_model('lstm_model_largest_parameters', 44, 4, 0.5, "squat")
