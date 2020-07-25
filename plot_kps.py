import urllib.parse
from pymongo import MongoClient
from data_preproccess.plot_body_part_positions import plot_body_part_positions

import numpy as np
import matplotlib.pyplot as plt
from list_manipulator import pop_all
from keypoints_extractor import KeypointsExtractor


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
        collection = db["push-up"]
        
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

def plot_all_videos(list_of_poses, body_part_index, body_part_name=None):
    if body_part_name is None:
        body_part_name = 'Body part'
    
    # Initialize plot list differences for each frame
    movements_x = []
    movements_y = []

    # nrows is to define the number of test data
    # ncols is to define the number plots for each coordinates (x & y)
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    fig.suptitle(f'{body_part_name} x and y Movements In each frame')

    # Define the matplot settings
    axs[0, 0].set_title(f'{body_part_name} x movements')
    axs[0, 0].set_xlim([1, 24])
    axs[0, 0].set_ylim([-1, 1])
    axs[0, 1].set_title(f'{body_part_name} y movements')
    axs[0, 1].set_xlim([1, 24])
    axs[0, 1].set_ylim([-1, 1])

    # Define the list of positive 
    # Or negative movements
    mov_pos_x = []
    mov_pos_y = []
    mov_neg_x = []
    mov_neg_y = []

    # Loop each frame
    for i, pose_frames in enumerate(list_of_poses):
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
        axs[0, 0].plot(frame_indexes, movements_x)
        
        # Plot the movement y coordinate of each video
        axs[0, 1].plot(frame_indexes, movements_y)
        
        pop_all(movements_x)
        pop_all(movements_y)

    for ax in axs.flat:
        ax.set(xlabel='Frames')

    # Print the threshold
    print('X axis positive movement threshold: ', str(sum(mov_pos_x)/len(mov_pos_x)))
    print('X axis negative movement threshold: ', str(sum(mov_neg_x)/len(mov_neg_x)))
    print('Y axis positive movement threshold: ', str(sum(mov_pos_y)/len(mov_pos_y)))
    print('Y axis negative movement threshold: ', str(sum(mov_neg_y)/len(mov_neg_y)))

    # Show plot
    plt.show()

def main():
    list_of_poses, list_of_labels = get_dataset()
    plot_all_videos(list_of_poses, NECK - 1, 'NECK')

main()