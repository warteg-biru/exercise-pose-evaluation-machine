import urllib.parse
from pymongo import MongoClient
from data_preproccess.plot_body_part_positions import plot_body_part_positions

import numpy as np
import matplotlib.pyplot as plt

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


def plot_all_videos(videos, body_part_index, body_part_name=None):
    if body_part_name is None:
        body_part_name = 'Body part'
    
    movements_x = []
    movements_y = []

    fig, axs = plt.subplots(nrows=12, ncols=2, squeeze=False)
    fig.suptitle(f'{body_part_name} x and y Position In each frame')
    for i, action in enumerate(videos):
        movements_x = [frame[body_part_index] for frame in action]
        movements_y = [frame[body_part_index + 1] for frame in action]
        frame_indexes = [idx + 1 for idx in range(len(action))]

        axs[i, 0].plot(frame_indexes, movements_x)
        axs[i, 0].set_title(f'{body_part_name} x position')
        
        axs[i, 1].plot(frame_indexes, movements_y)
        axs[i, 1].set_title(f'{body_part_name} y position')

    for ax in axs.flat:
        ax.set(xlabel='Frames')

    plt.show()

def main():
    list_of_poses, list_of_labels = get_dataset()
    plot_all_videos(list_of_poses, NECK, 'Neck')

main()