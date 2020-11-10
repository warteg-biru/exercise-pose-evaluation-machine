import warnings
warnings.simplefilter("ignore")

import os
import cv2
import random
import numpy as np
import collections
from datetime import datetime

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import LSTMCell, StackedRNNCells, RNN, Permute, Reshape, Dense, Dropout
from tensorflow.keras.optimizers import SGD


from list_manipulator import pop_all
from db_entity import get_initial_pose_dataset, get_starting_pose_binary_v2_from_db, get_starting_pose_binary_from_db
from keypoints_extractor import KeypointsExtractor

def validate_keypoints(keypoints):
    if len(keypoints) != 14:
        print("Data invalid! Expected 14 keypoints, received " + str(len(keypoints)) + ".")
        return False

    for index, coordinates in enumerate(keypoints):
        if index >= 0 or index <= 4:
            if coordinates[0] == 0 and coordinates[1] == 0:
                return False
        if len(coordinates) != 2:
            print("Data invalid! Expected 2 coordinates, received " + str(len(coordinates)) + ".")
            return False
    return True

def load_saved_model(model_path):
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model
    raise Exception('model does not exist')


class PoseDetector:
    def __init__(self, exercise_name):
        MODEL_PATH = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model/' + str(exercise_name) + '/' + str(exercise_name) + '_pose_model.h5'
        # MODEL_PATH = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model_v2/' + str(exercise_name) + '/' + str(exercise_name) + '_pose_model.h5'
        self.model = load_saved_model(MODEL_PATH)

    def predict(self, data):
        try:
            assert data.shape == (1, 28)
            prediction = self.model.predict(data)
            threshold = 0.9
            return 1 if prediction[0][0] > threshold else 0
        except Exception as e:
            print(e)

def test():
    pose_detector = PoseDetector("squat")
    kp_extractor = KeypointsExtractor()
    
    file_path = '/home/kevin/projects/initial-pose-data/images/v1/pos-negs/squat/pos/mirrored-VID_20200928_134747.mp4_186.jpg'
    # Opening OpenCV stream
    stream = cv2.VideoCapture(file_path)
    while True:
        try:
            ret, imageToProcess = stream.read()
        except Exception as e:
            # Break at end of frame
            break

        list_of_keypoints = kp_extractor.get_keypoints_and_id_from_img(imageToProcess)
        x = list_of_keypoints[0]
    keypoints = np.array(x).flatten()
    prediction = pose_detector.predict(np.array([keypoints]))
    print(prediction)

def train(exercise_name, dataset):
    # Initialize save path
    save_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model'
    # save_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model_v2'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += '/' + str(exercise_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += '/' + str(exercise_name) + '_pose_model.h5'

    # Get keypoint
    x = []
    y = []
    for data in dataset:
        keypoints = np.array(data["keypoints"]).flatten()
        x.append(keypoints)

        is_starting_pose = data["is_starting_pose"]
        label = 1 if is_starting_pose else 0
        y.append(label)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
    
    # Convert to np arrays so that we can use with TensorFlow
    x_train = np.array(x_train).astype(np.float32)
    x_test  = np.array(x_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)
    
    # Define number of features, labels, and hidden
    num_features = 28 # 14 pairs of (x, y) keypoints
    num_hidden = 8
    num_output = 1

    # Decaying learning rate
    learning_rate = 0.01
    lr_schedule = PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10,
        end_learning_rate= 0.00001
    )
    optimizer = SGD(learning_rate = lr_schedule)

    model = Sequential()
    model.add(Dense(60, input_shape=(num_features,)))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_output, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # Train model
    model.fit(x_train, y_train, epochs=250, batch_size=25, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.3)

    # Find accuracy
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    print("Class: " + exercise_name)

    # Save model to the designated path
    model.save(save_path)
    model.summary()
    print("Saved model")
        
if __name__ == '__main__':
    from multiprocessing import Process

    # Exercise Labels
    exercise_names = [
        # "sit-up"
        "push-up"
        # "plank"
        # "squat"
    ]

    # # Loop in each folder
    THREADS = []

    for exercise_name in exercise_names:
        # Get dataset
        dataset = get_starting_pose_binary_from_db(exercise_name)
        thread = Process(target=train, args=(exercise_name,dataset))
        thread.start()
        THREADS.append(thread)
    for t in THREADS:
        t.join()
    pop_all(THREADS)
    
    # variable = [[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0], [0.1186122208319354, 0.5149483570224415], [0.16306642523037668, 1.0], [0.414986984630671, 0.2859456585554382], [0.4152769395935169, 0.22839128934964859], [0.6964985413265004, 0.485291515118897], [0.9780199037363253, 0.5715992330523381], [0.4149430550319713, 0.37110755191467815], [0.7040218773323812, 0.5149120356887371], [1.0, 0.5146182490676815]]
    # test(variable)
    # dataset = get_starting_pose_binary_from_db("push-up")
    # for x in dataset:
    #     if not validate_keypoints(x["keypoints"]):
    #         print("Data is false")
    # test()