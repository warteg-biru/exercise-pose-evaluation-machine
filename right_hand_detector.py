import os

import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras

import sys
from sys import platform
import numpy as np
import os
import cv2
import tensorflow as tf
import collections
import matplotlib.pyplot as plt
import sys
from sys import platform
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from keypoints_extractor import get_upper_body_keypoints

if __name__ == '__main__':
    # Initialize paths
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine/right-hand-classification'
    checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up/right_hand_up.ckpt"

    # Get dataset folders
    dirs = os.listdir(base_path)

    x = []
    y = []
    # Loop in each folder
    for class_label, class_name in enumerate(dirs):
        class_dir = os.listdir(base_path+'/'+class_name)
        for file_name in class_dir:
            file_path = f'{base_path}/{class_name}/{file_name}'
            keypoints = get_upper_body_keypoints(file_path)

            x.append(np.array(keypoints).flatten())
            y.append([class_label])
    
    # Generate Training and Validation Sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # Convert to np arrays so that we can use with TensorFlow
    X_train = np.array(X_train).astype(np.float32)
    X_test  = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)

    # Define number of features, labels, and hidden
    num_features = 16
    num_labels = 1
    num_hidden = 5
    hidden_layers = num_features - 1
    
    '''
    build_model

    # Builds an ANN model for keypoint predictions
    @params {list of labels} image prediction labels to be tested
    @params {integer} number of features
    @params {integer} number of labels
    @params {integer} number of hidden layers
    '''
    graph = tf.Graph()
    with graph.as_default():
        # Initialize placeholder and constant
        tf_data = tf.placeholder(tf.float32, shape=(None, num_features), name= "tf_data")
        tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        
        # Note, since there is only 1 layer there are actually no hidden layers... but if there were
        # there would be num_hidden
        weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden]))
        weights_2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))

        # tf.zeros Automaticaly adjusts rows to input data batch size
        bias_1 = tf.Variable(tf.zeros([num_hidden]))
        bias_2 = tf.Variable(tf.zeros([num_labels]))
        
        logits_1 = tf.matmul(tf_data , weights_1 ) + bias_1
        rel_1 = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(rel_1, weights_2) + bias_2
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_2,  labels=tf_labels), name= "loss")
        optimizer = tf.train.GradientDescentOptimizer(.005, name="optimizer").minimize(loss)
        
        # Make prediction
        predict = tf.nn.sigmoid(logits_2, name = "predict")
        correct_pred = tf.equal(tf.round(predict), tf_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # Epochs
    num_steps = 5000
    # Run training session
    with tf.Session(graph = graph) as session:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        
        for step in range(num_steps):
            # Run training
            _,l, predictions, acc = session.run([optimizer, loss, predict, accuracy], feed_dict ={
                tf_data: X_train,
                tf_labels: y_train
            })
            
            if (step % 1000 == 0):
                # Print evaluation metrics
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % acc * 100)

                # Save checkpoints to file
                save_path = saver.save(session, checkpoint_path)