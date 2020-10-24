import warnings
warnings.simplefilter("ignore")

import os
import cv2
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
from db_entity import get_initial_pose_dataset
from keypoints_extractor import KeypointsExtractor

def load_saved_model(model_path):
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model
    raise Exception('model does not exist')


class PoseDetector:
    def __init__(self, exercise_name):
        MODEL_PATH = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model/' + str(exercise_name) + '/' + str(exercise_name) + '_pose_model.h5'
        self.model = load_saved_model(MODEL_PATH)

    def predict(self, data):
        try:
            assert data.shape == (1, 28)
            return self.model.predict(data)
        except Exception as e:
            print(e)

def test():
    pose_detector = PoseDetector("push-up")
    kp_extractor = KeypointsExtractor()
    
    file_path = '/home/kevin/projects/initial-pose-data/videos/push-up/push-up0.mp4'
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
        keypoints = np.array(x['Keypoints']).flatten()
        prediction = pose_detector.predict(np.array([keypoints]))
        print(np.argmax(prediction[0]))

def train(exercise_name, dataset):
    # Initialize save path
    save_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += '/' + str(exercise_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += '/' + str(exercise_name) + '_pose_model.h5'

    # Get keypoint
    x = []
    y = []
    for keypoint_name, keypoints in dataset.items():
        if keypoint_name == exercise_name:
            keypoints = [np.array(kp).flatten() for kp in keypoints]
            for kp in keypoints:
                x.append(kp)
                y.append(0)

    total_pos = len(y)
    neg_per_class = int(total_pos / 4)

    # Data label from mongodb
    for keypoint_name, keypoints in dataset.items():
        if keypoint_name != exercise_name:
            keypoints = keypoints if len(keypoints) < neg_per_class else keypoints[:neg_per_class]
            keypoints = [np.array(kp).flatten() for kp in keypoints]
            for kp in keypoints:
                x.append(kp)
                y.append(1)

    # Initialize paths
    base_path = "/home/kevin/projects/initial-pose-data/train_data"
    date_string = datetime.now().isoformat()
    filename = f'{exercise_name} binary pose k-fold results {date_string}'

    # Get dataset folders
    dirs = os.listdir(base_path)

    # One hot encoder
    y = np.array(y)
    # y = y.reshape(-1, 1)
    # one_hot = OneHotEncoder(sparse=False)
    # y = one_hot.fit_transform(y)

    # Split dataset
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)    
    
    # Define number of features, labels, and hidden
    num_features = 28 # 14 pairs of (x, y) keypoints
    num_hidden = 8
    num_labels = 5
    
    '''
    build_model

    # Builds an ANN model for keypoint predictions
    @params {list of labels} image prediction labels to be tested
    @params {integer} number of features
    @params {integer} number of labels as output
    @params {integer} number of hidden layers
    '''

    # Decaying learning rate
    learning_rate = 1e-2
    lr_schedule = PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10,
        end_learning_rate= 0.00001
    )
    optimizer = SGD(learning_rate = lr_schedule)

    model = Sequential()
    model.add(Dropout(0.2, input_shape=(num_features,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(num_hidden, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # Train model
    model.fit(x_train, y_train, epochs=100, batch_size=10, shuffle = True, validation_data = (x_test, y_test), validation_split = 0.3)

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
        "sit-up",
        "push-up",
        "plank",
        "squat"
    ]
    # Get dataset
    dataset = get_initial_pose_dataset()

    # Loop in each folder
    THREADS = []

    for exercise_name in exercise_names:
        thread = Process(target=train, args=(exercise_name,dataset))
        thread.start()
        THREADS.append(thread)
    for t in THREADS:
        t.join()
    pop_all(THREADS)