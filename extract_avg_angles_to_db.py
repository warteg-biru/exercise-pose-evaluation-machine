import os
import statistics
from db_entity import DBEntity
from keypoints_extractor import\
    normalize_keypoints_for_plot_kps, KeypointsExtractor,\
    make_min_max_scaler, normalize_keypoints_from_external_scaler
from kp_index import NOSE, NECK, R_SHOULDER, R_ELBOW, R_WRIST,\
L_SHOULDER, L_ELBOW, L_WRIST, MID_HIP, R_HIP, R_KNEE, R_ANKLE,\
L_HIP, L_KNEE, L_ANKLE, R_EYE, L_EYE, R_EAR, L_EAR, L_BIG_TOE,\
L_SMALL_TOE, L_HEEL, R_BIG_TOE, R_SMALL_TOE, R_HEEL
from plot_exercise_movement_angle import get_average_angles_from_poses
from angle_calculator import AngleCalculator

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

# Nge tes
def __get_average_angles(to_be_averaged):
    minimum_length_of_arr = len(to_be_averaged[0])
    # filter array yang panjang nya tidak sama
    to_be_averaged = [arr for arr in to_be_averaged if len(arr) == minimum_length_of_arr]
    result = [statistics.mean(k) for k in zip(*to_be_averaged)]
    return result


def get_average_angles_from_poses(list_of_poses):
    to_be_averaged = []
    for pose_frames in list_of_poses:
        angles_list = get_angles_from_frames(pose_frames)
        to_be_averaged.append(angles_list)
    return __get_average_angles(to_be_averaged)


def extract_avg_angles_to_db(videos_dir, class_name):
    video_paths = [f'{videos_dir}/{file}' for file in os.listdir(videos_dir) ]
    video_paths = video_paths[:1]
    # Initialize keypoints extractor
    kp_extractor = KeypointsExtractor()
    # Set from 
    sets = []
    all_exercise_reps = []
    all_exercise_x_low = []
    all_exercise_y_low = []
    for video_path in video_paths:
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
    db_entity = DBEntity()
    average_angles = get_average_angles_from_poses(normalized_sets)
    db_entity.insert_average_angles(average_angles, class_name)


if __name__ == "__main__":
    situp_dir = '/home/kevin/projects/test_plot_kps_videos/sit-up'
    extract_avg_angles_to_db(situp_dir, 'sit-up')