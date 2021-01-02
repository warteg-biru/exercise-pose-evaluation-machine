import warnings
warnings.simplefilter("ignore")

import os
import sys
import cv2
import numpy as np
import collections
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from list_manipulator import pop_all
import traceback

# Get the local openpose path
sys.path.append('/usr/local/python')

try:
    from openpose import pyopenpose as op
except Exception as e:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?'
    )

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.tools.generate_detections import create_box_encoder

import gc

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

# Normalize keypoints
@params{list of keypoints} Array of keypoints
@params{integer} lowest x coordinate
@params{integer} lowest y coordinate
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
        print(end="")

    # normalize data between 0 to 1
    scaler = MinMaxScaler()
    scaler.fit(nom_keypoint)
    ret_val = scaler.transform(nom_keypoint)
    return ret_val


'''
normalize_keypoints_for_plot_kps

# Normalize keypoints purposefully for keypoint plotting
@params{list of frames} Array of frames
@params{integer} lowest x coordinate
@params{integer} lowest y coordinate
'''
# Get normalized keypoints
def normalize_keypoints_for_plot_kps(keypoint_frames, x_low, y_low):
    # Define normalized keypoints array
    all_keypoints = []
    # Define normalized keypoints array for each frames
    nom_keypoint_frame = []
    try:
        for i, _ in enumerate(keypoint_frames):
            # Define temporary array for normalized keypoints 
            # (To be appended to normalized frame keypoints array)
            nom_keypoint = []
            for count, kp in enumerate(_):
                # Avoid x=0 and y=0 because some keypoints that are not discovered
                # If x=0 and y=0 than it doesn't need to be substracted with the
                # lowest x and y points
                if kp['x'] != 0 and kp['y'] != 0:
                    x = kp['x'] - x_low
                    y = kp['y'] - y_low
                    nom_keypoint.append([x,y])
                    all_keypoints.append([x,y])
                else:
                    nom_keypoint.append([kp['x'], kp['y']])
                    all_keypoints.append([kp['x'], kp['y']])
            
            # Add keypoints for the selected frame to the array
            nom_keypoint_frame.append(nom_keypoint)
    except Exception as e:
        print(end="")
        
    # Normalize data between 0 to 1
    scaler = MinMaxScaler(feature_range=(0,1))

    # Fit the scaler according to all the keypoints
    # From all the frames
    scaler.fit(all_keypoints)

    # Define MinMax'ed keypoints for each frame
    min_max_keypoint_frame = []
    for x in nom_keypoint_frame:
        ret_val = scaler.transform(x) # ret_val is an array of keypoints (in other words Frames)
        min_max_keypoint_frame.append(ret_val)

    # Return normalized keypoint frames
    return min_max_keypoint_frame


'''
make_min_max_scaler

# Make min max scaler out of the coordinates from all frames
@params{list of frames} Array of frames
@params{integer} lowest x coordinate
@params{integer} lowest y coordinate
'''
# Get normalized keypoints
def make_min_max_scaler(keypoint_frames, x_low, y_low):
    # Define normalized keypoints array
    all_keypoints = []
    # Define normalized keypoints array for each frames
    nom_keypoint_frame = []
    try:
        for i, _ in enumerate(keypoint_frames):
            # Define temporary array for normalized keypoints 
            # (To be appended to normalized frame keypoints array)
            nom_keypoint = []
            for count, kp in enumerate(_):
                # Avoid x=0 and y=0 because some keypoints that are not discovered
                # If x=0 and y=0 than it doesn't need to be substracted with the
                # lowest x and y points
                if kp['x'] != 0 and kp['y'] != 0:
                    x = kp['x'] - x_low
                    y = kp['y'] - y_low
                    nom_keypoint.append([x,y])
                    all_keypoints.append([x,y])
                else:
                    nom_keypoint.append([kp['x'], kp['y']])
                    all_keypoints.append([kp['x'], kp['y']])
            
            # Add keypoints for the selected frame to the array
            nom_keypoint_frame.append(nom_keypoint)
    except Exception as e:
        print(end="")
        
    # Normalize data between 0 to 1
    scaler = MinMaxScaler(feature_range=(0,1))

    # Fit the scaler according to all the keypoints
    # From all the frames
    scaler.fit(all_keypoints)

    # Return normalized keypoint frames
    return scaler


'''
normalize_keypoints_from_external_scaler

# Normalize keypoints using an external scaler
@params{list of frames} Array of frames
@params{scaler} min max scaler
'''
# Get normalized keypoints
def normalize_keypoints_from_external_scaler(keypoint_frames, scaler):
    # Define normalized keypoints array for each frames
    nom_keypoint_frame = []
    try:
        for i, _ in enumerate(keypoint_frames):
            # Define temporary array for normalized keypoints 
            # (To be appended to normalized frame keypoints array)
            nom_keypoint = []
            for count, kp in enumerate(_):
                nom_keypoint.append([kp['x'], kp['y']])
            
            # Add keypoints for the selected frame to the array
            nom_keypoint_frame.append(nom_keypoint)
    except Exception as e:
        print(end="")

    # Define MinMax'ed keypoints for each frame
    min_max_keypoint_frame = []
    for x in nom_keypoint_frame:
        ret_val = scaler.transform(x) # ret_val is an array of keypoints (in other words Frames)
        min_max_keypoint_frame.append(ret_val)

    # Return normalized keypoint frames
    return min_max_keypoint_frame

'''
write_text

# Write text to image
@params {object} image
@params {string} text
@params {tuple} coordinates
'''
def write_text(img, text, org):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (0, 0, 255)
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method 
    return cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

class KeypointsExtractor:
    def __init__(self):
        params = set_params()
        # Constructing OpenPose object allocates GPU memory
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        self.opWrapper = opWrapper
        # Load Deep SORT model
        self.model_path = './deep_sort/model_data/mars-small128.pb'
        self.nms_max_overlap = 1.0
        self.encoder = create_box_encoder(self.model_path, batch_size=1)

    '''
    get_upper_body_keypoints_and_id_from_img

    # Get keypoints from the image
    @params {object} image
    '''
    # Scan person for keypoints and ID
    def get_upper_body_keypoints_and_id_from_img(self, img):
        # KP ordering of body parts
        NOSE        = 0
        NECK        = 1
        R_SHOULDER  = 2
        R_ELBOW     = 3
        R_WRIST     = 4
        L_SHOULDER  = 5
        L_ELBOW     = 6
        L_WRIST     = 7
        MID_HIP     = 8
        R_EYE       = 15
        L_EYE       = 16
        R_EAR       = 17
        L_EAR       = 18

        # Define bodyparts to get the selected keypoints
        BODY_PARTS = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR, NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP]

        # Set tracker
        max_cosine_distance = 0.2
        nn_budget = 100
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        # Get data points (datum)
        datum = op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop([datum])

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
                features = self.encoder(datum.cvOutputData, boxes)

                # For a non-empty item add to the detection array
                def nonempty(xywh): return xywh[2] != 0 and xywh[3] != 0
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(
                    boxes, features) if nonempty(bbox)]

                # Run non-maxima suppression.
                np_boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    np_boxes, self.nms_max_overlap, scores)
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
        except Exception as e:
            print(end="")

        return list_of_pose_and_id, datum.cvOutputData
        

    '''
    get_keypoints_and_id_from_img

    # Get keypoints from the image
    @params {object} image
    '''
    # Scan image for keypoints
    def get_keypoints_and_id_from_img(self, img):
        # KP ordering of body parts
        NECK        = 1
        R_SHOULDER  = 2
        R_ELBOW     = 3
        R_WRIST     = 4
        L_SHOULDER  = 5
        L_ELBOW     = 6
        L_WRIST     = 7
        MID_HIP     = 8
        R_HIP       = 9
        R_KNEE      = 10
        R_ANKLE     = 11
        L_HIP       = 12
        L_KNEE      = 13
        L_ANKLE     = 14

        # Define bodyparts to get the selected keypoints
        BODY_PARTS = [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE]

        # Set tracker
        max_cosine_distance = 0.2
        nn_budget = 100
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        # Get data points (datum)
        datum = op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop([datum])
        img_show = datum.cvOutputData

        # Initialize lists
        arr = []
        boxes = []
        list_of_pose_temp = []
        list_of_pose_and_id = []
        print_keypoints = []
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

                # # Normalize keypoint
                normalized_keypoints = normalize_keypoints(arr, x_low, y_low)
                print_keypoints.append((int((x_high - x_low) / 2) + x_low, y_low))
                list_of_pose_temp.append(normalized_keypoints)

                # Make the box
                boxes.append([x_low, y_low, width, height])

                # Encode the features inside the designated box
                features = self.encoder(datum.cvOutputData, boxes)

                # For a non-empty item add to the detection array
                def nonempty(xywh): return xywh[2] != 0 and xywh[3] != 0
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(
                    boxes, features) if nonempty(bbox)]

                # Run non-maxima suppression.
                np_boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    np_boxes, self.nms_max_overlap, scores)
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
                        img_show = write_text(img_show, f'person_id {track.track_id}', print_keypoints[track_idx])

            return list_of_pose_and_id, img_show
        except Exception as e:
            print(e)
            print(end="")


    '''
    get_keypoints_and_id_from_img_without_normalize

    # Get keypoints from the image
    @params {image} image data
    '''
    # Scan image for keypoints
    def get_keypoints_and_id_from_img_without_normalize(self, img):
        # KP ordering of body parts
        NECK        = 1
        R_SHOULDER  = 2
        R_ELBOW     = 3
        R_WRIST     = 4
        L_SHOULDER  = 5
        L_ELBOW     = 6
        L_WRIST     = 7
        MID_HIP     = 8
        R_HIP       = 9
        R_KNEE      = 10
        R_ANKLE     = 11
        L_HIP       = 12
        L_KNEE      = 13
        L_ANKLE     = 14

        # Define bodyparts to get the selected keypoints
        BODY_PARTS = [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE]

        # Set tracker
        max_cosine_distance = 0.2
        nn_budget = 100
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        # Get data points (datum)
        datum = op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop([datum])

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

                # # Normalize keypoint
                list_of_pose_temp.append(arr)

                # Make the box
                boxes.append([x_low, y_low, width, height])

                # Encode the features inside the designated box
                features = self.encoder(datum.cvOutputData, boxes)

                # For a non-empty item add to the detection array
                def nonempty(xywh): return xywh[2] != 0 and xywh[3] != 0
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(
                    boxes, features) if nonempty(bbox)]

                # Run non-maxima suppression.
                np_boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    np_boxes, self.nms_max_overlap, scores)
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
            print(end="")

    '''
    scan_image

    # Get keypoints from the image
    @params {string} image path
    '''
    # Scan image for keypoints
    def scan_image(self, img_path):
        # Opening image in OpenCV
        imageToProcess = cv2.imread(img_path)

        # Get data points (datum)
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum])

        # Get output image
        output_image = datum.cvOutputData

        # Define new array
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
            print(end="")


    '''
    scan_image_without_normalize

    # Get keypoints from the image
    @params {string} image path
    '''
    # Scan image for keypoints
    def scan_image_without_normalize(self, img_path):
        # Opening image in OpenCV
        imageToProcess = cv2.imread(img_path)

        # Get data points (datum)
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum])

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

            return arr, x_low, y_low, output_image
        except Exception as e:
            print(end="")


    '''
    scan_video

    # Get keypoints from the video
    @params {string} video path
    @params {integer} class of exercise
    '''
    # Scan video for keypoints
    def scan_video(self, video_path, keypoints_to_extract):
        # Opening OpenCV stream
        stream = cv2.VideoCapture(video_path)

        # Set font
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
            self.opWrapper.emplaceAndPop([datum])

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
                        # Check which keypoints to extract
                        if count in keypoints_to_extract:
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

                    # Normalize keypoints by frame
                    normalized_keypoint = normalize_keypoints(arr, x_low, y_low)
                    list_of_pose.append(normalized_keypoint)
            except Exception as e:
                print(end="")

        return list_of_pose


    '''
    scan_video_without_normalize

    # Get keypoints from the video
    @params {string} video path
    @params {integer} class of exercise
    '''
    # Scan video for keypoints
    def scan_video_without_normalize(self, video_path, keypoints_to_extract):
        # Opening OpenCV stream
        stream = cv2.VideoCapture(video_path)

        # Define list of pose, x low, and y low
        list_of_pose = []
        list_of_x_low = []
        list_of_y_low = []
        while True:
            try:
                # Stream
                ret, imageToProcess = stream.read()
                datum = op.Datum()
                datum.cvInputData = imageToProcess
            except Exception as e:
                # Break at end of frame
                break

            # Find keypoints
            self.opWrapper.emplaceAndPop([datum])

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
                        # Check which keypoints to extract
                        if count in keypoints_to_extract:
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

                    # Append list of pose, x low, and y low
                    list_of_pose.append(arr)
                    list_of_x_low.append(x_low)
                    list_of_y_low.append(y_low)
            except Exception as e:
                print(end="")

        return list_of_pose, list_of_x_low, list_of_y_low


    '''
    get_upper_body_keypoints_and_id

    # Get keypoints from the image
    @params {string} image path
    '''
    # Scan person for keypoints and ID
    def get_upper_body_keypoints_and_id(self, img_path):
        # KP ordering of body parts
        NOSE        = 0
        NECK        = 1
        R_SHOULDER  = 2
        R_ELBOW     = 3
        R_WRIST     = 4
        L_SHOULDER  = 5
        L_ELBOW     = 6
        L_WRIST     = 7
        MID_HIP     = 8
        R_EYE       = 15
        L_EYE       = 16
        R_EAR       = 17
        L_EAR       = 18

        BODY_PARTS = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR, NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP]

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
        self.opWrapper.emplaceAndPop([datum])

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
                features = self.encoder(datum.cvOutputData, boxes)

                # For a non-empty item add to the detection array
                def nonempty(xywh): return xywh[2] != 0 and xywh[3] != 0
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(
                    boxes, features) if nonempty(bbox)]

                # Run non-maxima suppression.
                np_boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    np_boxes, self.nms_max_overlap, scores)
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
            print(end="")
            

    '''
    get_upper_body_keypoint

    # Get keypoints from the image
    @params {string} image path
    '''
    def get_upper_body_keypoint(self, image_path):
        # Get keypoints from image
        keypoints = self.scan_image(image_path)

        # KP ordering of body parts
        NOSE        = 0
        NECK        = 1
        R_SHOULDER  = 2
        R_ELBOW     = 3
        R_WRIST     = 4
        L_SHOULDER  = 5
        L_ELBOW     = 6
        L_WRIST     = 7
        MID_HIP     = 8
        R_EYE       = 15
        L_EYE       = 16
        R_EAR       = 17
        L_EAR       = 18

        # Define bodyparts to get the selected keypoints
        BODY_PARTS = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR, NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP]

        # Define keypoints array to be returned
        new_keypoints = []
        for index, keypoint in enumerate(keypoints):
            if index in BODY_PARTS:
                new_keypoints.append(keypoint)
        
        return new_keypoints


    '''
    get_min_max

    # Get min max coordinates
    @params {keypoints} keypoints
    '''
    def get_min_max(self, keypoints):
        x_min = 9999999
        y_min = 9999999
        x_max = 0
        y_max = 0
        for keypoint in keypoints:
            x_min = keypoint['x'] if keypoint['x'] < x_min else x_min 
            x_max = keypoint['x'] if keypoint['x'] > x_max else x_max
            y_min = keypoint['y'] if keypoint['y'] < y_min else y_min
            y_max = keypoint['y'] if keypoint['y'] > y_max else y_max
        return x_min, y_min, x_max, y_max

        
    '''
    get_min_max_frames

    # Get min max frames
    @params {frames} frames
    '''
    def get_min_max_frames(self, frames):
        x_min = 9999999
        y_min = 9999999
        x_max = 0
        y_max = 0
        for keypoints in frames:
            kp_x_min, kp_y_min, kp_x_max, kp_y_max = self.get_min_max(keypoints)
            x_min = kp_x_min if kp_x_min < x_min else x_min 
            x_max = keypoint['x'] if keypoint['x'] > x_max else x_max
            y_min = kp_y_min if kp_y_min < y_min else y_min
            y_max = kp_y_max if kp_y_max > y_max else y_max

        return x_min, y_min, x_max, y_max
        
    '''
    get_bounded_coordinates

    # Get keypoints from the image
    @params {string} image path
    '''
    def get_bounded_coordinates(self, prediction, imageToProcess):
        # Define min-max and length of the person
        kp = self.get_keypoints_and_id_from_img_without_normalize(imageToProcess)
        x_min, y_min, x_max, y_max = self.get_min_max(kp[0]['Keypoints'])
        x_length = x_max - x_min
        y_length = y_max - y_min

        # Get image dimensions
        dimensions = imageToProcess.shape
        width = dimensions[1]
        height = dimensions[0]

        # Make padding according to exercise type
        if prediction != "squat":
            x_min = x_min - (x_length * 50 / 100) if x_min - (x_length * 50 / 100) >= 0  else 0
            x_max = x_max + (x_length * 50 / 100) if x_max + (x_length * 50 / 100) <= width else width
            y_min = y_min - y_length if y_min - y_length >= 0 else 0
            y_max = y_max + y_length if y_max + y_length <= height else height
        else:
            x_min = x_min - x_length if x_min - x_length >= 0  else 0
            x_max = x_max + x_length if x_max + x_length <= width else width
            y_min = y_min - y_length if y_min - y_length >= 0 else 0
            y_max = y_max + y_length if y_max + y_length <= height else height

        # Return bounded coordinates
        return x_min, y_min, x_max, y_max