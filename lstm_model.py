import collections
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Enable eager execution
tf.enable_eager_execution()

# Determines the number of data the dataset divides into
batch_size = 64

# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Determines each input sequence which will be of size (28, 28) (height is treated like time).
input_dim = 28

# Output with the amount of units
units = 16

# Labels are from 0 to 2
output_size = 3

from numpy import array

# Get dataset
x, y = get_dataset()
x = pad_sequences(x)
x = array(x)

# Pop all in array
def pop_all(l):
    r, l[:] = l[:], []
    return r

reshaped_sample = []
reshaped_frames = []

for i, sample in enumerate(x):
    pop_all(reshaped_frames)
    for j, frames in enumerate(sample):
        frames = np.concatenate(frames).ravel()
        reshaped_frames.append(frames)
        
    reshaped_sample.append(reshaped_frames)

x = array(reshaped_sample)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
  # CuDNN is only available at the layer level, and not at the cell level.
  # This means `LSTM(units)` will use the CuDNN kernel,
  # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
  if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN.
    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(66, 75))
  else:
    # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))

  # Define sequential model
  model = tf.keras.models.Sequential([
        lstm_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size)]
  )
  return model

# Build model
model = build_model(allow_cudnn_kernel=True)

# Compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer='sgd',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=175)

sample, sample_label = x_test[0], y_test[0]

with tf.device('CPU:0'):
  cpu_model = build_model(allow_cudnn_kernel=True)
  cpu_model.set_weights(model.get_weights())
  result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
  print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
  plt.imshow(sample, cmap=plt.get_cmap('gray'))