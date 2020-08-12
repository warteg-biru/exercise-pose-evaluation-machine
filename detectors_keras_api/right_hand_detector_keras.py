import os
import cv2
import sys
import numpy as np
import collections

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

def get_dataset():
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine/not_important_folder/right-hand-classification'
    # Get dataset folders
    dirs = os.listdir(base_path)
    x = []
    y = []
    kp_extractor = KeypointsExtractor()
    # Loop in each folder
    for class_label, class_name in enumerate(dirs):
        class_dir = os.listdir(base_path+'/'+class_name)
        for file_name in class_dir:
            file_path = f'{base_path}/{class_name}/{file_name}'
            keypoints = kp_extractor.get_upper_body_keypoint(file_path)
            x.append(np.array(keypoints).flatten())
            y.append([class_label])

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
def create_model():
    # Define number of features, labels, and hidden
    num_features = 16
    num_labels = 1
    num_hidden = 5
    hidden_layers = num_features - 1

    model = Sequential()
    model.add(Dropout(0.2, input_shape=(num_features,)))
    model.add(Dense(num_hidden, activation='relu'))
    model.add(Dense(num_labels, activation='sigmoid'))

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
        MODEL_PATH = 'models/right_hand_up/right_hand_up.h5'
        self.model = load_saved_model(MODEL_PATH)
    
    '''
    predict
    @param {np.array} data - 1 row matrix of upper body keypoints with shape (1, 16)
    '''
    def predict(self, data):
        try:
            assert data.shape == (1, 16)
            prediction = self.model.predict(data)
            # prediction < 0.35 == right hand up
            # 0.35 >= prediction == not right hand up
            return 1 if prediction[0][0] < 0.35 else 0
        except Exception as e:
            print(e)



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset()
    MODEL_PATH = 'models/right_hand_up/right_hand_up.h5'
    
    try:
        # Try load model
        model = load_saved_model(MODEL_PATH)
        model.summary()

        single_correct_label = y_test[15]
        single_test_data = np.array([X_test[15]])
        prediction = model.predict(single_test_data)

        print('model prediction is ', 1 if prediction[0][0] > 0.5 else 0)
        print('correct label should be ', single_correct_label)
    except Exception as e:
        # If no model, create new model, train, and save
        model = create_model()
        model.summary()
        model.fit(X_train, y_train, epochs=500, batch_size=10)
        _, accuracy = model.evaluate(X_train, y_train)
        model.save(MODEL_PATH)
        print(e)