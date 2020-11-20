import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.simplefilter("ignore")

import cv2
import sys
import numpy as np
import collections

ROOT_DIR = "/home/kevin/projects/exercise_pose_evaluation_machine"
sys.path.append(ROOT_DIR)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.tools.generate_detections import create_box_encoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor
from db_entity import get_right_hand_up_dataset

def get_dataset():
    x = []
    y = []

    # Get dataset
    true_dataset = get_right_hand_up_dataset(True)
    false_dataset = get_right_hand_up_dataset(False)

    # Use the same amount of data for true and false
    true_len = len(true_dataset)
    false_len = len(false_dataset)
    max_len = true_len if true_len > false_len else false_len
    true_dataset = true_dataset[:max_len]
    false_dataset = false_dataset[:max_len]

    # Loop in each folder
    for keypoints in true_dataset:
        x.append(np.array(keypoints).flatten())
        y.append(1)
    for keypoints in false_dataset:
        x.append(np.array(keypoints).flatten())
        y.append(0)

    # Generate Training and Validation Sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # Convert to np arrays so that we can use with TensorFlow
    X_train = np.array(X_train).astype(np.float32)
    X_test  = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)

    return X_train, X_test, y_train, y_test

'''
build_model

# Builds an ANN model for keypoint predictions
@params {list of labels} image prediction labels to be tested
@params {integer} number of features
@params {integer} number of labels
@params {integer} number of hidden layers
'''
def create_model(double):
    # Define number of features, labels, and hidden
    num_features = 26
    num_output = 1

    model = Sequential()
    model.add(Dense(60, input_shape=(num_features,)))
    model.add(Dense(30, activation='relu'))
    if double:
        model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_output, activation='sigmoid'))

    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_saved_model(model_path):
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model
    raise Exception('model does not exist')


class RightHandUpDetector:
    def __init__(self):
        MODEL_PATH = '/home/kevin/projects/exercise_pose_evaluation_machine/models/right_hand_up/right_hand_up.h5'
        self.model = load_saved_model(MODEL_PATH)
    
    '''
    predict
    @param {np.array} data - 1 row matrix of upper body keypoints with shape (1, 26)
    '''
    def predict(self, data):
        try:
            assert data.shape == (1, 26)
            prediction = self.model.predict(data)
            # print("prediction: ", prediction)
            # prediction > threshold == right hand up
            # threshold >= prediction == not right hand up
            return 1 if prediction[0][0] > 0.90 else 0
        except Exception as e:
            print(e)



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset()
    MODEL_PATH = 'models/right_hand_up/right_hand_up.h5'
    
    try:
        # Try load model
        model = load_saved_model(MODEL_PATH)
        model.summary()

        print(len(X_test))
        data_idx = 20
        single_correct_label = y_test[data_idx]
        single_test_data = np.array([X_test[data_idx]])
        prediction = model.predict(single_test_data)
        print("prediction score is: ", prediction)
        print('model prediction is ', 1 if prediction[0][0] > 0.5 else 0)
        print('correct label should be ', single_correct_label)
    except Exception as e:
        # If no model, create new model, train, and save
        print(f'{e}, training a new model')
        model = create_model(True)
        model.summary()
        model.fit(X_train, y_train, epochs=150, batch_size=50)
        _, accuracy = model.evaluate(X_train, y_train)
        model.save(MODEL_PATH)
