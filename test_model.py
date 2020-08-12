import os
import cv2
import time
import collections
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keypoints_extractor import KeypointsExtractor, normalize_keypoints, normalize_keypoints_from_external_scaler, make_min_max_scaler
from predict_init_pose import predict_initial_pose

import tensorflow as tf

if __name__ == '__main__':
    # Base paths
    checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/lstm_model/"
    base_path = "/home/kevin/projects/push-up.mp4"
    kp_extractor = KeypointsExtractor()
    
    # with tf.Session() as session:
    session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    session.run(init)

    # Load checkpoint model
    loader = tf.train.import_meta_graph(checkpoint_path+'lstm_model.ckpt.meta')
    loader.restore(session, tf.train.latest_checkpoint(checkpoint_path))

    # Get required tensors
    predict = session.graph.get_tensor_by_name('predict:0')

    # Get placeholder and keypoint data
    tf_data = session.graph.get_tensor_by_name('tf_data:0')

    # Opening OpenCV stream
    all_exercise_reps, all_exercise_x_low, all_exercise_y_low = kp_extractor.scan_video_without_normalize(base_path, [1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))
    normalized_reps = normalize_keypoints_from_external_scaler(all_exercise_reps, scaler)

    INPUT_SIZE = 24
    input_buffer = []
    for idx, list_of_keypoints in enumerate(normalized_reps):
        print("Initial pose is " + str(predict_initial_pose(
            normalize_keypoints(
                all_exercise_reps[idx],
                all_exercise_x_low[idx],
                all_exercise_y_low[idx]
            )
        )))
        time.sleep(1)
        try: 
            # Transform keypoints list to array
            keypoints = np.array(list_of_keypoints).flatten().astype(np.float32)
            input_buffer.append(keypoints)
            if (len(input_buffer) > INPUT_SIZE):
                input_buffer = input_buffer[1:]
        except Exception as e:
            # Break at end of frame
            print(e)
            pass
        if len(input_buffer) == INPUT_SIZE:
            # Get prediction
            predictions = session.run(predict, feed_dict={tf_data: [input_buffer]})
            print(np.argmax(predictions))