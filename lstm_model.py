import collections
import matplotlib.pyplot as plt
import numpy as np
from numpy import array

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import urllib.parse
from pymongo import MongoClient

# Get from Mongo DB
def get_dataset():
    # Initialize temp lists
    list_of_poses = []
    list_of_labels = []

    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))


        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db["test"]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # try catch for MongoDB insert
        try:
            for x in collection.find():
                list_of_poses.append(x["list_of_pose"])
                list_of_labels.append(x["exercise_type"])
        except Exception as e:
            print("Failed to get data from database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 
    
    return list_of_poses, list_of_labels

# Pop all from array
def pop_all(l):
    r, l[:] = l[:], []
    return r

# Label encoder
def one_hot(y_):
    # One hot encoding of the network outputs
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

# Load the networks inputs
def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]], 
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)
    
    X_ = np.array(np.split(X_,blocks))

    return X_ 

# Load the networks outputs
def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # for 0-based indexing 
    return y_ - 1

# Define labels
LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
] 

# Labels
output_size = len(LABELS)

DATASET_PATH = "/home/kevin/learn/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input/data/HAR_pose_activities/database/"

X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"

y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 32 # 32 timesteps per series

# Enable eager execution
tf.enable_eager_execution()

# Determines the number of data the dataset divides into
batch_size = 12

# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Determines each input sequence which will be of size (28, 28) (height is treated like time).
input_dim = 28

# Output with the amount of units
units = 16

# Load dataset
x_train = load_X(X_train_path)
x_test = load_X(X_test_path)

y_train = one_hot(load_y(y_train_path))
y_test = one_hot(load_y(y_test_path))

input_shape = x_train.shape

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
  # CuDNN is only available at the layer level, and not, MaxPool#d at the cell level.
  # This means `LSTM(units)` will use the CuDNN kernel,
  # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
  if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN.
    lstm_layer = tf.keras.layers.LSTM(units, input_shape = (input_shape[1], input_shape[2]))
  else:
    # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))

  # Define sequential model
  model = tf.keras.models.Sequential([
        lstm_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size)
    ]
  )
  return model

# Build model
model = build_model(allow_cudnn_kernel=True)

# Show model summary
print(model.summary)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.7)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            optimizer=optimizer,
            metrics=['accuracy'])

# Train model
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=100)

sample, sample_label = x_test[0], y_test[0]

with tf.device('CPU:0'):
  cpu_model = build_model(allow_cudnn_kernel=True)
  cpu_model.set_weights(model.get_weights())
  result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
  print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
  plt.imshow(sample, cmap=plt.get_cmap('gray'))
