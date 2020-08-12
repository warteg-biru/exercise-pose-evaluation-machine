import os
import sys
import cv2
import numpy as np
import collections

import tensorflow as tf

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor

if __name__ == '__main__':
    # Initialize paths
    base_path = '/home/kevin/projects/Exercise Starting Pose'
    checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.ckpt"

    # Get dataset folders
    dirs = os.listdir(base_path)

    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()

    x = []
    y = []
    # Loop in each folder
    for class_label, class_name in enumerate(dirs):
        class_dir = os.listdir(base_path+'/'+class_name)
        for file_name in class_dir:
            file_path = f'{base_path}/{class_name}/{file_name}'
            image = cv2.imread(file_path)
            list_of_pose_and_id = kp_extractor.get_keypoints_and_id_from_img(image)
            keypoints = list_of_pose_and_id[0]['Keypoints']

            x.append(np.array(keypoints).flatten())
            y.append(class_label)
    
    # One hot encoder
    y = np.array(y)
    y = y.reshape(-1, 1)
    one_hot = OneHotEncoder(sparse=False)
    y = one_hot.fit_transform(y)
    
    # Generate Training and Validation Sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # Convert to np arrays so that we can use with TensorFlow
    X_train = np.array(X_train).astype(np.float32)
    X_test  = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)

    # Define number of features, labels, and hidden
    num_features = 28 # 14 pairs of (x, y) keypoints
    num_labels = 4
    num_hidden = 10
    
    
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
        tf_labels = tf.constant(y_train)
        
        # Note, since there is only 1 layer there are actually no hidden layers... 
        # but if there were there would be num_hidden
        weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden]))
        weights_2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))

        # tf.zeros Automaticaly adjusts rows to input data batch size
        bias_1 = tf.Variable(tf.zeros([num_hidden]))
        bias_2 = tf.Variable(tf.zeros([num_labels]))
        
        # Make neural network
        logits_1 = tf.matmul(tf_data , weights_1 ) + bias_1
        rel_1 = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(rel_1, weights_2) + bias_2
        
        # Apply softmax cross entropy as loss and gradient descent optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2,  labels=tf_labels), name= "loss")
        optimizer = tf.train.GradientDescentOptimizer(.005, name="optimizer").minimize(loss)
        
        # Make prediction
        predict = tf.nn.softmax(logits_2, name = "predict")

    # Define accuracy
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

    # Epochs
    num_steps = 10000

    # Run training session
    with tf.Session(graph = graph) as session:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        
        for step in range(num_steps):
            # Run training
            _,l, predictions = session.run([optimizer, loss, predict], feed_dict ={
                tf_data: X_train
            })
            
            if (step % 2000 == 0):
                # Print evaluation metrics
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy( predictions, y_train[:, :]))

                # Save checkpoints to file
                save_path = saver.save(session, checkpoint_path)