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

def gram_schmidt():
    import numpy

    def gs_cofficient(v1, v2):
        return numpy.dot(v2, v1) / numpy.dot(v1, v1)

    def multiply(cofficient, v):
        return map((lambda x : x * cofficient), v)

    def proj(v1, v2):
        return multiply(gs_cofficient(v1, v2) , v1)

    def gs(X):
        Y = []
        for i in range(len(X)):
            temp_vec = X[i]
            for inY in Y :
                proj_vec = proj(inY, X[i])
                #print "i =", i, ", projection vector =", proj_vec
                temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
                #print "i =", i, ", temporary vector =", temp_vec
            Y.append(temp_vec)
        return Y

    test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
    test2 = numpy.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])

    print(numpy.array(gs(test)))
    print(numpy.array(gs(test2)))

# Scan image for keypoints
def scan_image(img_path):
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

    # Opening image in OpenCV
    imageToProcess = cv2.imread(img_path)

    # Get data points (datum)
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display the stream
    output_image = imageToProcess
    arr = []
    boxes = []

    res = main(datum.poseKeypoints[0])
    
    try:
        # Get highest and lowest keypoints
        for count, x in enumerate(datum.poseKeypoints[0]):
            kp1 = int((x[0]*1)+300)
            kp2 = int((x[1]*1)+300)
            kp3 = int((x[2]*1)+300)
            print(kp1, kp2, kp3)

            # Radius of circle 
            radius = 20
            
            # Blue color in BGR 
            color = (255, 0, 0) 
            
            # Line thickness of 2 px 
            thickness = 2
            output_image = cv2.circle(output_image, (x[0], x[1]), radius, (0,255,0), thickness)
            output_image = cv2.circle(output_image, (kp1, kp2), radius, color, thickness) 
    except Exception as e:
        print(e)

    # Output image
    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", output_image)
    cv2.waitKey(0)

# L-20 MCS 507 Fri 11 Oct 2013 : gramschmidt.py

"""
Given pseudo code for the Gram-Schmidt method,
define Python code.
"""

import numpy as np

def gramschmidt(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
    return Q, R

def main(A):
    """
    Prompts for n and generates a random matrix.
    """
    Q, R = gramschmidt(A)
    return Q

scan_image('/home/kevin/Downloads/batfleck.jpg')