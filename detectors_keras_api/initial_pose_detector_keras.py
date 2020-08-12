import os
import sys
import cv2
import numpy as np
import collections

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor

def load_saved_model(model_path):
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model
    raise Exception('model does not exist')


class InitialPoseDetector:
    def __init__(self):
        MODEL_PATH = 'models/initial_pose_model/initial_pose_model.h5'
        self.model = load_saved_model(MODEL_PATH)

    def predict(self, data):
        try:
            assert data.shape == (1, 28)
            prediction = self.model.predict(data)
            return self.get_exercise_name_from_prediction(prediction)
        except Exception as e:
            print(e)

    '''
    Result lookup

    #plank -> 0
    #situps -> 1
    #dumbell-curl -> 2
    #pushup -> 3
    '''
    def get_exercise_name_from_prediction(self, prediction):
        res_lookup = ['plank', 'sit-up', 'dumbell-curl', 'push-up']
        # prediction shape is (1, 4)
        exercise_index = np.argmax(prediction[0])
        return res_lookup[exercise_index]

        try:
            assert data.shape == (1, 28)
            prediction = self.model.predict(data)
            return self.get_exercise_name_from_prediction(prediction)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # Initialize paths
    base_path = '/home/kevin/projects/Exercise Starting Pose'
    save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5"

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
    num_hidden = 8
    num_labels = 4
    
    '''
    build_model

    # Builds an ANN model for keypoint predictions
    @params {list of labels} image prediction labels to be tested
    @params {integer} number of features
    @params {integer} number of labels as output
    @params {integer} number of hidden layers
    '''
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(num_features,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(num_hidden, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=200, batch_size=10)

    # Find accuracy
    _, accuracy = model.evaluate(X_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))

    # Get keypoints from image
    img_test_path = "/home/kevin/projects/Exercise Starting Pose/Dumbell Curl/dumbellcurl1.jpg"
    image = cv2.imread(img_test_path)
    list_of_pose_and_id = kp_extractor.get_keypoints_and_id_from_img(image)
    keypoints = list_of_pose_and_id[0]['Keypoints']

    # Prepare keypoints to feed into network
    keypoints = np.array(keypoints).flatten().astype(np.float32)

    '''
    Result lookup

    #plank -> 0
    #situps -> 1
    #dumbell-curl -> 2
    #pushup -> 3
    '''
    res_lookup = ['plank', 'sit-up', 'dumbell-curl', 'push-up']
    print("Result is " + res_lookup[
        np.argmax(
            model.predict(
                np.array(
                    [
                        keypoints
                    ]
                )
            )
        )
    ])

    # Save model to the designated path
    model.save(save_path)
    print("Saved model")