import os

import cv2
import numpy as np


from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

import sys
from sys import platform

from pymongo import MongoClient
import urllib.parse

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/usr/local/python')

try:
    from openpose import pyopenpose as op
except:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

'''
set_params

# Set the openpose parameters
'''
# Set openpose default parameters
def set_params():
    params = dict()
    '''
         params untuk menambah performance
        '''
    params["net_resolution"] = "320x176"
    params["face_net_resolution"] = "320x320"
    params["model_pose"] = "BODY_25"

    # params["logging_level"] = 3
    # params["output_resolution"] = "-1x-1"
    # params["net_resolution"] = "-1x368"
    # params["alpha_pose"] = 0.6
    # params["scale_gap"] = 0.3
    # params["scale_number"] = 1
    # params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    # params["num_gpu_start"] = 0
    # params["disable_blending"] = False
    # Ensure you point to the correct path where models are located

    params["model_folder"] = './openpose/models'
    return params

'''
pop_all

# Pop entire list
@params {list} list to be popped
'''
# Pop all in array
def pop_all(l):
    r, l[:] = l[:], []
    return r

'''
normalize_keypoints

# normalize the gathered keypoints between 1 and 0
@params {list} keypoints
@params {float} lowest bound x
@params {float} lowest bound y
'''
# Get normalized keypoints
def normalize_keypoints(keypoint, x_low, y_low):
    # Define normalized keypoints array
    nom_keypoint = []
    try:
        normalized_image = ""
        for count, kp in enumerate(keypoint):
            # Avoid x=0 and y=0 because some keypoints that are not discovered
            # If x=0 and y=0 than it doesn't need to be substracted with the
            # lowest x and y points
            if kp['x'] != 0 and kp['y'] != 0:
                x = kp['x'] - x_low
                y = kp['y'] - y_low
                nom_keypoint.append([x,y])
            else:
                nom_keypoint.append([kp['x'], kp['y']])

    except Exception as e:
        print(str(e))
        
    # normalize data between 0 to 1
    scaler = MinMaxScaler()
    scaler.fit(nom_keypoint)
    ret_val = scaler.transform(nom_keypoint)
    return ret_val

'''
scan_video

# Get keypoints from the video
@params {string} video path
@params {integer} class of exercise
'''
# Scan video for keypoints
def scan_video(video_path, class_type):
    model_path = './deep_sort/model_data/mars-small128.pb'
    params = set_params()
    nms_max_overlap = 1.0

    # Constructing OpenPose object allocates GPU memory
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Set tracker
    max_cosine_distance = 0.2
    nn_budget = 100
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Initialize encoder
    encoder = create_box_encoder(model_path, batch_size=1)

    # Opening OpenCV stream
    stream = cv2.VideoCapture(video_path)

    # Set font
    font = cv2.FONT_HERSHEY_SIMPLEX
    list_of_pose = []
    while True:
        try:
            ret, imageToProcess = stream.read()
            datum = op.Datum()
            datum.cvInputData = imageToProcess
        except Exception as e:
            # Break at end of frame
            break

        # Find keypoints
        opWrapper.emplaceAndPop([datum])

        # Get output image processed by Openpose
        output_image = datum.cvOutputData
        
        # Define keypoints array and binding box array
        arr = []
        boxes = []
        
        try:
            # Loop each of the 17 keypoints
            for keypoint in datum.poseKeypoints:
                pop_all(arr)
                x_high = 0
                x_low = 9999
                y_high = 0
                y_low = 9999

                # Get highest and lowest keypoints
                for count, x in enumerate(keypoint):
                    # Avoid x=0 and y=0 because some keypoints that are not discovered.
                    # This "if" is to define the LOWEST and HIGHEST discovered keypoint.
                    if x[0] != 0 and x[1] != 0:
                        if x_high < x[0]:
                            x_high = x[0]
                        if x_low > x[0]:
                            x_low = x[0]
                        if y_high < x[1]:
                            y_high = x[1]
                        if y_low > x[1]:
                            y_low = x[1]
                            
                    # Add pose keypoints to a dictionary
                    KP = {
                        'x': x[0],
                        'y': x[1]
                    }
                    # Append dictionary to array
                    arr.append(KP)

                # Find the highest and lowest position of x and y 
                # (Used to draw rectangle)
                if y_high - y_low > x_high - x_low:
                    height = y_high-y_low
                    width = x_high-x_low
                else:
                    height = x_high-x_low
                    width = y_high-y_low

                # Draw rectangle (get width and height)
                y_high = int(y_high + height / 40)
                y_low = int(y_low - height / 12)
                x_high = int(x_high + width / 5)
                x_low = int(x_low - width / 5)

                # Normalize keypoint
                normalized_keypoint = normalize_keypoints(arr, x_low, y_low)
                list_of_pose.append(normalized_keypoint)

                # Make the box
                boxes.append([x_low, y_low, width, height])

                # Encode the features inside the designated box
                features = encoder(output_image, boxes)

                # For a non-empty item add to the detection array
                def nonempty(xywh): return xywh[2] != 0 and xywh[3] != 0
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(
                    boxes, features) if nonempty(bbox)]

                # Run non-maxima suppression.
                np_boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    np_boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Update tracker.
                tracker.predict()
                tracker.update(detections)

                # Draw rectangle and put text
                for track in tracker.tracks:
                    bbox = track.to_tlwh()
                    cv2.rectangle(output_image, (x_high, y_high),
                                  (x_low, y_low), (255, 0, 0), 2)
                    cv2.putText(output_image, "id%s - ts%s" % (track.track_id, track.time_since_update),
                                (int(bbox[0]), int(bbox[1])-20), 0, 5e-3 * 100, (0, 255, 0), 2)
        except Exception as e:
            # That means there's an error
            print("Error")
            print(str(e))
            # break
            pass

        # Display the stream
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", output_image)
        key = cv2.waitKey(1)

        # Quit
        if key == ord('q'):
            break

    # Insert to mongodb
    insert_array_to_db(list_of_pose,class_type)

    # Release stream and destroy all windows
    stream.release()
    cv2.destroyAllWindows()


'''
insert_array_to_db

Insert array of poses into Mongo DB
@params {list} list_of_pose
@params {float} class of exercise
'''
def insert_array_to_db(list_of_pose, class_type):
    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db["test"]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # Initialize temp lists
        list_of_pose_arr = []
        list_of_kp_arr = []
        # try catch for MongoDB insert
        try:
            # list_of_pose consists of poses which is type of np array
            # Loop to get np array inside a normal array 
            # (mongodb does not accept np array)
            for pose in list_of_pose:
                list_of_pose_arr.append(pose.tolist())
                
            # Make new object
            pose = {
                "list_of_pose": list_of_pose_arr,
                "exercise_type": class_type
            }

            # Insert into database collection
            rec_id1 = collection.insert_one(pose) 
            print("Data inserted with record ids",rec_id1) 
        except Exception as e:
            print("Failed to insert data to database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 

'''
get_dataset

# Get data from MongoDB
'''
# Get from Mongo DB
def get_dataset():
    # Initialize temp lists
    list_of_poses = []
    list_of_labels = []

    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db["test"]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # try catch for MongoDB insert
        try:
            for x in collection.find():
                list_of_poses.append(x["list_of_pose"])
                list_of_labels.append(x["exercise_type"])
        except Exception as e:
            print("Failed to get data from database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 
    
    return list_of_poses, list_of_labels
  

if __name__ == '__main__':
    # # Define base path for the dataset
    base_path = '/home/kevin/projects/dataset_exercise_pose_evaluation_machine2'
    

    # # Get dataset folders
    dirs = os.listdir(base_path)

    # # Loop in each folder
    for idx, d in enumerate(dirs):
        folder_path = base_path+'/'+d
        files = os.listdir(folder_path)

        # Scan every file
        for f in files:
            file_path = folder_path+'/'+f
            scan_video(file_path, idx)
    # get_array_from_db()