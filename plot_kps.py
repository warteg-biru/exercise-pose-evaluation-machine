import os
import numpy as np
import urllib.parse
from pymongo import MongoClient
import matplotlib.pyplot as plt
from list_manipulator import pop_all
from sklearn.preprocessing import MinMaxScaler
from data_preproccess.plot_body_part_positions import plot_body_part_positions
from keypoints_extractor import normalize_keypoints_for_plot_kps, KeypointsExtractor
from kp_index import get_kp_index_by_int


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


'''
make_min_max_scaler
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
        print(str(e))
        
    # Normalize data between 0 to 1
    scaler = MinMaxScaler(feature_range=(0,1))

    # Fit the scaler according to all the keypoints
    # From all the frames
    scaler.fit(all_keypoints)

    # Return normalized keypoint frames
    return scaler


'''
normalize_keypoints_from_external_scaler
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
        print(str(e))

    # Define MinMax'ed keypoints for each frame
    min_max_keypoint_frame = []
    for x in nom_keypoint_frame:
        print(x)
        ret_val = scaler.transform(x) # ret_val is an array of keypoints (in other words Frames)
        min_max_keypoint_frame.append(ret_val)

    # Return normalized keypoint frames
    return min_max_keypoint_frame

def setup_plot(axs, idx, body_part_name):
    # Define the matplot settings
    axs[idx, 0].set_title(f'{body_part_name} x movements')
    axs[idx, 0].set_xlim([1, 24])
    axs[idx, 0].set_ylim([-1, 1])
    axs[idx, 1].set_title(f'{body_part_name} y movements')
    axs[idx, 1].set_xlim([1, 24])
    axs[idx, 1].set_ylim([-1, 1])


# Plot the keypoints
def plot_keypoints(list_of_poses, body_part_indexes, body_part_name=None):
    # Default body part name if body_part_name is None
    if body_part_name is None:
        body_part_name = 'Body part'

    # nrows is to define the number of test data
    # ncols is to define the number plots for each coordinates (x & y)
    fig, axs = plt.subplots(nrows=len(body_part_indexes), ncols=2, squeeze=False)
    fig.suptitle(f'{body_part_name} x and y Movements In each frame')
    
        
    # Initialize plot list differences for each frame
    movements_x = []
    movements_y = []

    # Define the list of positive 
    # Or negative movements
    mov_pos_x = []
    mov_pos_y = []
    mov_neg_x = []
    mov_neg_y = []

    # Loop each frame
    for i, pose_frames in enumerate(list_of_poses):
        for j, body_part_index in enumerate(body_part_indexes):
            setup_plot(axs, j, get_kp_index_by_int(body_part_index))
            for idx, action in enumerate(pose_frames):
                if idx < len(pose_frames) - 1:
                        # Get the plotted movements between the two keypoints
                        # For each movement at the x axis in body_part_index keypoints add to movements_x
                        mov_x = pose_frames[idx+1][body_part_index][0] - pose_frames[idx][body_part_index][0]
                        if mov_x > 0:
                            mov_pos_x.append(mov_x)
                        else:
                            mov_neg_x.append(mov_x)
                        movements_x.append(mov_x)
                        # For each movement at the y axis in body_part_index keypoints add to movements_y
                        mov_y = pose_frames[idx+1][body_part_index][1] - pose_frames[idx][body_part_index][1]
                        if mov_y > 0:
                            mov_pos_y.append(mov_y)
                        else:
                            mov_neg_y.append(mov_y)
                        movements_y.append(mov_y)

            # Get the total number of frames
            frame_indexes = [idx + 1 for idx in range(len(pose_frames) - 1)]
            
            # Plot the movement x coordinate of each video
            axs[j, 0].plot(frame_indexes, movements_x)
            
            # Plot the movement y coordinate of each video
            axs[j, 1].plot(frame_indexes, movements_y)
            
            pop_all(movements_x)
            pop_all(movements_y)

    for ax in axs.flat:
        ax.set(xlabel='Frames')

    mov_pos_x_count = len(mov_pos_x) if len(mov_pos_x) > 0 else 1
    mov_neg_x_count = len(mov_neg_x) if len(mov_neg_x) > 0 else 1
    mov_pos_y_count = len(mov_pos_y) if len(mov_pos_y) > 0 else 1
    mov_neg_y_count = len(mov_neg_y) if len(mov_neg_y) > 0 else 1

    # Print the threshold
    print('X axis positive movement threshold: ', str(sum(mov_pos_x)/mov_pos_x_count))
    print('X axis negative movement threshold: ', str(sum(mov_neg_x)/mov_neg_x_count))
    print('Y axis positive movement threshold: ', str(sum(mov_pos_y)/mov_pos_y_count))
    print('Y axis negative movement threshold: ', str(sum(mov_neg_y)/mov_neg_y_count))

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

    scaler = make_min_max_scaler(all_exercise_reps, min(all_exercise_x_low), min(all_exercise_y_low))

    normalized_sets = []
    for x in sets:
        normalized_sets.append(normalize_keypoints_from_external_scaler(x, scaler))

    # Plot keypoints
    body_parts_to_show = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    plot_keypoints(normalized_sets, body_parts_to_show, 'NECK')