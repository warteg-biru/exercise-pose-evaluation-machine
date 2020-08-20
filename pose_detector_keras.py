import os
import cv2
import numpy as np
import collections

# import sys
# sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor

def load_saved_model(model_path):
    if os.path.isfile(model_path):
        model = load_model(model_path)
        return model
    raise Exception('model does not exist')


class PoseDetector:
    def __init__(self, exercise_name):
        MODEL_PATH = 'models/pose_model/' + str(exercise_name) + '/' + str(exercise_name) + '_pose_model.h5'
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

def train():
    # Initialize base path
    base_path = '/home/kevin/projects/initial-pose-data/pos-negs'

    # Get dataset folders
    dirs = os.listdir(base_path)

    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()

    # Loop in each folder
    for class_name in dirs:
        # Initialize save path
        save_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/pose_model'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path += '/' + str(class_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path += '/' + str(class_name) + '_pose_model.h5'
        
        x = []
        y = []
        class_dir = os.listdir(base_path+'/'+class_name)
        for pos_neg_folder in class_dir:
            pos_neg_dir = os.listdir(base_path+'/'+class_name+'/'+pos_neg_folder)
            for file_name in pos_neg_dir:
                file_path = f'{base_path}/{class_name}/{pos_neg_folder}/{file_name}'
                image = cv2.imread(file_path)
                list_of_pose_and_id = kp_extractor.get_keypoints_and_id_from_img(image)
                keypoints = list_of_pose_and_id[0]['Keypoints']

                x.append(np.array(keypoints).flatten())
                y.append(pos_neg_folder)
    
        # Label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        y = label_encoder.transform(y)
        
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
        num_labels = 2
        
        '''
        build_model

        # Builds an ANN model for keypoint predictions
        @params {list of labels} image prediction labels to be tested
        @params {integer} number of features
        @params {integer} number of labels as output
        @params {integer} number of hidden layers
        '''
        model = Sequential()
        # model.add(Dropout(0.2, input_shape=(num_features,)))
        model.add(Dense(12, input_dim=num_features, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_hidden, activation='relu'))
        model.add(Dense(num_labels, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        
        # Train model
        model.fit(X_train, y_train, epochs=100, batch_size=10)

        # Find accuracy
        _, accuracy = model.evaluate(X_test, y_test)
        print('Accuracy: %.2f' % (accuracy*100))
        print("Class: " + class_name)

        # Save model to the designated path
        model.save(save_path)
        model.summary()
        print(label_encoder.classes_)
        print("Saved model")
        
if __name__ == '__main__':
    test()