import os
import numpy as np
import urllib.parse
from pymongo import MongoClient
import matplotlib.pyplot as plt
from list_manipulator import pop_all
from sklearn.preprocessing import MinMaxScaler
from angle_calculator import AngleCalculator

from kp_index import get_kp_index_by_int

from keypoints_extractor import normalize_keypoints_for_plot_kps, KeypointsExtractor, make_min_max_scaler, normalize_keypoints_from_external_scaler

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
R_HIP       = 9
R_KNEE      = 10
R_ANKLE     = 11
L_HIP       = 12
L_KNEE      = 13
L_ANKLE     = 14
R_EYE       = 15
L_EYE       = 16
R_EAR       = 17
L_EAR       = 18
L_BIG_TOE   = 19
L_SMALL_TOE = 20
L_HEEL      = 21
R_BIG_TOE   = 22
R_SMALL_TOE = 23
R_HEEL      = 24

# Setup plot
def setup_plot(cells, idx, exercise_name, xlim=[1, 24], ylim=None):
    # Define the matplot settings
    cells[idx, 0].set_title(f'{exercise_name} key angles')
    cells[idx, 0].set_xlim(xlim)
    if ylim is not None:
        cells[idx, 0].set_ylim(ylim)

def get_angle_from_keypoints(keypoints):
    # print(keypoints)
    kp_neck = keypoints[NECK]
    kp_mid_hip = keypoints[MID_HIP]
    kp_left_knee = keypoints[L_KNEE]
    angle_calc = AngleCalculator()
    angle = angle_calc.get_angle_from_three_keypoints(kp_neck, kp_left_knee, kp_mid_hip)
    return angle

def get_angles_from_frames(pose_frames):
    angles_list = []
    for idx, action in enumerate(pose_frames):
        angle = get_angle_from_keypoints(action)
        angles_list.append(angle)
    return angles_list

# Plot the keypoints
def plot_angles(list_of_poses, num_videos, exercise_name):
    fig, cells = plt.subplots(nrows=num_videos, ncols=1, squeeze=False)
    fig.suptitle(f'{exercise_name} key angles in each frame')

    # Loop each frame
    for i, pose_frames in enumerate(list_of_poses):
        setup_plot(cells, i, 'Sit Up', xlim=[1, 48])
        # get all the angles
        angles_list = get_angles_from_frames(pose_frames)
        # Get the total number of frames
        frame_indexes = [idx + 1 for idx in range(len(pose_frames))]
        # Plot the change in angles over the frames
        cells[i, 0].plot(frame_indexes, angles_list)

    for cell in cells.flat:
        cell.set(xlabel='Frames')
    # Show plot
    plt.show()

def plot_angles_in_one_plot(list_of_poses, exercise_name):
    fig, cells = plt.subplots(nrows=1, ncols=1, squeeze=False)
    fig.suptitle(f'{exercise_name} key angle in each frame')
    # Loop each frame
    for pose_frames in list_of_poses:
        setup_plot(cells, 0, 'Sit Up', xlim=[1, 48])
        # get all the angles
        angles_list = get_angles_from_frames(pose_frames)
        # Get the total number of frames
        frame_indexes = [idx + 1 for idx in range(len(pose_frames))]
        # Plot the change in angles over the frames
        cells[0, 0].plot(frame_indexes, angles_list)
    for cell in cells.flat:
        cell.set(xlabel='Frames')
    # Show plot
    plt.show()


if __name__ == '__main__':
    # Initialize video path
    push_up_dir = '/home/kevin/projects/test_plot_kps_videos/push-up'
    plank_dir = '/home/kevin/projects/test_plot_kps_videos/plank'
    situp_dir = '/home/kevin/projects/test_plot_kps_videos/sit-up'
    # video_paths = [f'{push_up_dir}/{file}' for file in os.listdir(push_up_dir)]
    # video_paths = [f'{plank_dir}/{file}' for file in os.listdir(plank_dir) ]
    video_paths = [f'{situp_dir}/{file}' for file in os.listdir(situp_dir) ]

    # Set from 
    sets = []
    all_exercise_reps = []
    all_exercise_x_low = []
    all_exercise_y_low = []
    for video_path in video_paths:
        # Initialize keypoints extractor
        kp_extractor = KeypointsExtractor()
        # Get keypoints from video
        exercise_reps, exercise_x_low, exercise_y_low = kp_extractor.scan_video_without_normalize(video_path, [NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE])
        # Normalize pose keypoints based on entire list of keypoints
        for rep in exercise_reps:
            all_exercise_reps.append(rep)
        all_exercise_x_low.append(min(exercise_x_low))
        all_exercise_y_low.append(min(exercise_y_low))
        # Add to sets
        sets.append(exercise_reps)

    # Make min max scaler and save
    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))

    normalized_sets = []
    for x in sets:
        normalized_sets.append(normalize_keypoints_from_external_scaler(x, scaler))
    # Plot angles
    num_videos = len(video_paths)
    body_parts_to_show = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    # plot_angles(normalized_sets, num_videos, 'Sit Up')
    plot_angles_in_one_plot(normalized_sets, 'Sit Up')