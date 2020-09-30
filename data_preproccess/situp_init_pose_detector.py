import os
import cv2
import numpy as np
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')
from keypoints_extractor import KeypointsExtractor

class SitUpInitialPoseDetector():
    def __init__(self):
        self.MODEL_PATH\
            = "/home/kevin/projects/exercise_pose_evaluation_machine/models/situp_initial_pose_model/situp_initial_pose_model.h5"
        self.model = self.load_saved_model(self.MODEL_PATH)

    def load_saved_model(self, model_path):
        if os.path.isfile(model_path):
            model = load_model(model_path)
            return model
        raise Exception('model does not exist')

    def predict(self, data):
        try:
            assert data.shape == (1, 28)
            prediction = self.model.predict(data)
            return np.squeeze(prediction)
        except Exception as e:
            print(e)


def get_files_and_labels():
    pos_dir = "/home/kevin/projects/situp_init_pose_data/pos"
    neg_dir = "/home/kevin/projects/situp_init_pose_data/neg"

    # X - files
    pos_files = [
        os.path.join(pos_dir, filename) for filename in os.listdir(pos_dir)
    ]
    neg_files = [
        os.path.join(neg_dir, filename) for filename in os.listdir(neg_dir)[:len(pos_files)]
    ]
    # y - labels
    pos_labels = [1 for _ in pos_files]
    neg_labels = [0 for _ in neg_files]

    files = pos_files + neg_files
    labels = pos_labels + neg_labels

    return files, labels



def get_keypoints_from_files(kp_extractor, files):
    kps = []
    for f_path in files:
        image = cv2.imread(f_path)
        list_of_pose_and_id = kp_extractor.get_keypoints_and_id_from_img(image)
        keypoints = list_of_pose_and_id[0]['Keypoints']
        kps.append(np.array(keypoints).flatten())
    return kps


def format_data_to_np_array(X_train, X_test, y_train, y_test):
    # Convert to np arrays so that we can use with TensorFlow
    X_train = np.array(X_train).astype(np.float32)
    X_test  = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)
    
    return X_train, X_test, y_train, y_test


def build_model():
    # Define number of features, labels, and hidden
    num_features = 28 # 14 pairs of (x, y) keypoints
    num_hidden = 8
    num_labels = 1
    
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
    model.add(Dense(num_labels, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


def train(MODEL_PATH, kp_extractor):
    files, labels = get_files_and_labels()
    X = get_keypoints_from_files(kp_extractor, files)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    X_train, X_test, y_train, y_test = format_data_to_np_array(
        X_train, X_test, y_train, y_test
        )

    model = build_model()
    # Train model
    model.fit(X_train, y_train, epochs=200, batch_size=10)

    # Find accuracy
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save(MODEL_PATH)



def test(MODEL_PATH, kp_extractor):
    model = load_model(MODEL_PATH)

    test_pos_dir = pos_dir = "/home/kevin/projects/situp_init_pose_data/test_data/pos"
    test_neg_dir = pos_dir = "/home/kevin/projects/situp_init_pose_data/test_data/neg"
    test_pos_files = [
        os.path.join(test_pos_dir, filename) for filename in os.listdir(test_pos_dir)
    ]
    test_neg_files = [
        os.path.join(test_neg_dir, filename) for filename in os.listdir(test_neg_dir)
    ]

    pos_keypoints = get_keypoints_from_files(kp_extractor, test_pos_files)
    neg_keypoints = get_keypoints_from_files(kp_extractor, test_neg_files)

    for X in pos_keypoints:
        X = np.array([X]).astype(np.float32)
        pred = model.predict(X)
        print(pred)
        print("should be close to 1")

    for X in neg_keypoints:
        X = np.array([X]).astype(np.float32)
        pred = model.predict(X)
        print(pred)
        print("sould be close to 0")



if __name__ == "__main__":
    MODEL_PATH\
        = "/home/kevin/projects/exercise_pose_evaluation_machine/models/situp_initial_pose_model/situp_initial_pose_model.h5"
    kp_extractor = KeypointsExtractor()

    if os.path.isfile(MODEL_PATH) is False:
        train(MODEL_PATH, kp_extractor)


    test(MODEL_PATH, kp_extractor)