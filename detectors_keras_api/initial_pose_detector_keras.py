import os
import cv2
import traceback
import numpy as np
import collections
import time

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

import sys
sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from db_entity import get_initial_pose_dataset

def load_saved_model(model_path):
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model
    raise Exception('model does not exist')


class InitialPoseDetector:
    def __init__(self):
        MODEL_PATH = '/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5'
        self.model = load_saved_model(MODEL_PATH)

    def predict(self, data):
        try:
            assert data.shape == (1, 28)
            prediction = self.model.predict(data)
            return self.get_exercise_name_from_prediction(prediction)
        except BaseException as e:
            traceback.print_exc()

    '''
    Result lookup

    #plank -> 0
    #situps -> 1
    #dumbell-curl -> 2
    #pushup -> 3
    '''
    def get_exercise_name_from_prediction(self, prediction):
        res_lookup = ['plank', 'push-up', 'sit-up', 'squat']
        exercise_index = np.argmax(prediction[0])
        # print(prediction[0][exercise_index])
        # import time
        # time.sleep(2000)
        if prediction[0][exercise_index] < 0.6:
            return -1
        return res_lookup[exercise_index]
       


def train():
    # Initialize paths
    base_path = "/home/kevin/projects/initial-pose-data/train_data"
    save_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/initial_pose_model/initial_pose_model.h5"

    # Get dataset folders
    dirs = os.listdir(base_path)

    exercise_name_labels = { "sit-up": 0, "plank": 1, "squat": 2, "push-up": 3, "stand": 4 }
    x = []
    y = []
    dataset = get_initial_pose_dataset()
    for exercise_name, keypoints in dataset.items():
        for kp in keypoints:
            x.append(kp)
            y.append(exercise_name_labels[exercise_name])

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
    num_labels = 5
    
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
    model.fit(X_train, y_train, epochs=200, batch_size=10, shuffle = True, validation_data = (X_test, y_test), validation_split = 0.3))

    # Find accuracy
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    # Get keypoints from image
    img_test_path = "/home/kevin/projects/initial-pose-data/images/pos-negs/push-up/pos/push-up47.mp4_24.jpg"
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
    res_lookup = ['plank', 'push-up', 'sit-up', 'squat']
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


def test(test_file_path):
    initial_pose_detector = InitialPoseDetector()
    kp_extractor = KeypointsExtractor()
    # Opening OpenCV stream
    stream = cv2.VideoCapture(test_file_path)
    while True:
        # Stream --
        try:
            ret, imageToProcess = stream.read()
        except Exception as e:
            # Break at end of frame
            print(e)
            break
        cv2.imshow("Initail Pose Detector Test", imageToProcess)
        key = cv2.waitKey(1)
        # Get keypoints and predict
        list_of_keypoints = kp_extractor.get_keypoints_and_id_from_img(imageToProcess)
        try: 
            for x in list_of_keypoints:
                # Transform keypoints list to array
                keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                keypoints = np.array([keypoints])
                # Get prediction
                prediction = initial_pose_detector.predict(keypoints)
                found_exercise = prediction
                print("Initial pose prediction result: " + str(prediction))
        except Exception as e:
            print(e)



if __name__ == '__main__':
    # test_file_path = "/home/kevin/projects/dataset-handsup-to-exercise/pushup.mp4.mp4"
    # test(test_file_path)
    train()