import os

import cv2
import numpy as np

import sys
from sys import platform
import numpy as np
import os
import cv2
import collections
import matplotlib.pyplot as plt
import sys
from sys import platform
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder
from sklearn.preprocessing import MinMaxScaler

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/usr/local/python')

try:
    from openpose import pyopenpose as op
except:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

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
    params["model_folder"] = './openpose/models'
    return params

'''
normalize_keypoints
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
scan_image

# Get keypoints from the image
@params {string} image path
'''
# Scan image for keypoints
def scan_image(img_path):
    model_path = './deep_sort/model_data/mars-small128.pb'
    params = set_params()

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

    # Opening image in OpenCV
    imageToProcess = cv2.imread(img_path)

    # Get data points (datum)
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Get output image
    output_image = datum.cvOutputData

    arr = []
    try:
        pop_all(arr)
        x_high = 0
        x_low = 9999
        y_high = 0
        y_low = 9999

        # Get highest and lowest keypoints
        for count, x in enumerate(datum.poseKeypoints[0]):
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
        normalized_keypoints = normalize_keypoints(arr, x_low, y_low)

        return normalized_keypoints
    except Exception as e:
        print(e)

        
'''
get_upper_body_keypoints_and_id

# Get keypoints from the image
@params {string} image path
'''
# Scan person for keypoints and ID
def get_upper_body_keypoints_and_id(img_path):
    # KP ordering of body parts
    NECK        = 1
    R_SHOULDER  = 2
    R_ELBOW     = 3
    R_WRIST     = 4
    L_SHOULDER  = 5
    L_ELBOW     = 6
    L_WRIST     = 7
    MID_HIP     = 8

    BODY_PARTS = [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP]

    # Constructing OpenPose object allocates GPU memory
    params = set_params()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Load Deep SORT model
    model_path = './deep_sort/model_data/mars-small128.pb'
    nms_max_overlap = 1.0
    encoder = create_box_encoder(model_path, batch_size=1)

    # Set tracker
    max_cosine_distance = 0.2
    nn_budget = 100
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Opening image in OpenCV
    imageToProcess = cv2.imread(img_path)

    # Get data points (datum)
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Initialize lists
    arr = []
    boxes = []
    list_of_pose_temp = []
    list_of_pose_and_id = []
    try:

        # Get highest and lowest keypoints
        for kp_idx, keypoint in enumerate(datum.poseKeypoints):
            pop_all(arr)
            x_high = 0
            x_low = 9999
            y_high = 0
            y_low = 9999

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
                if count in BODY_PARTS:
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
            normalized_keypoints = normalize_keypoints(arr, x_low, y_low)
            list_of_pose_temp.append(normalized_keypoints)

            # Make the box
            boxes.append([x_low, y_low, width, height])

            # Encode the features inside the designated box
            features = encoder(datum.cvOutputData, boxes)

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

            # Make pose and person ID list
            if kp_idx == len(datum.poseKeypoints)-1:
                for track_idx, track in enumerate(tracker.tracks):
                    bbox = track.to_tlwh()
                    list_of_pose_and_id.append({
                        "Keypoints": list_of_pose_temp[track_idx],
                        "ID": track.track_id
                    })

        return list_of_pose_and_id
    except Exception as e:
        print(e)
    
'''
get_upper_body_keypoint

# Get keypoints from the image
@params {string} image path
'''
def get_upper_body_keypoint(image_path):
    keypoints = scan_image(image_path)
    # KP ordering of body parts
    NECK        = 1
    R_SHOULDER  = 2
    R_ELBOW     = 3
    R_WRIST     = 4
    L_SHOULDER  = 5
    L_ELBOW     = 6
    L_WRIST     = 7
    MID_HIP     = 8

    BODY_PARTS = [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP]

    new_keypoints = []
    for index, keypoint in enumerate(keypoints):
        if index in BODY_PARTS:
            new_keypoints.append(keypoint)
    
    return new_keypoints