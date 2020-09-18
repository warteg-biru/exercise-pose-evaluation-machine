import sys
import math

sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine/')

from keypoints_extractor import pop_all
from db_entity import get_count, get_dataset_with_limit, insert_array_to_db


# Define class types for each exercise  
CLASS_TYPE = [
    "push-up",
    "sit-up"
    # "plank",
    # "squat",
    # "dumbell-curl",
]

def get_class_type_frame_length(class_type):
    return 48 if class_type == "sit-up" else 24

def double_array_in_array(array):
    new_array = []
    array_i_array = []
    for x in array:
        # Append the array twice
        array_i_array.append(x)
        array_i_array.append(x)

        # Append to new array
        new_array.append(array_i_array)
        # Pop placeholder array
        pop_all(array_i_array)

    return new_array

def halve_array_in_array(array):
    new_array = []
    array_i_array = []
    for x in array:
        # Append the halved array
        half_length = math.ceil(len(x) / 2)
        len(x[0:half_length])
        array_i_array.append(x[0:half_length])

        # Append to new array
        new_array.append(array_i_array)
        # Pop placeholder array
        pop_all(array_i_array)

    return new_array

'''
make_inverse_dataset

# Makes an inverse dataset consisting of keypoints from other exercises
@params {string} class name
'''
def make_inverse_dataset(class_type):
    # Define the total amount of data to get
    total_count = 0
    count_per_class = []
    class_type_count = get_count(class_type)
    list_of_poses = []

    # Get inverse dataset count for each class type
    for x in CLASS_TYPE:
        if x != class_type:
            curr_count = get_count(x)
            total_count+=curr_count
            count_per_class.append({
                'count': curr_count,
                'class_type': x
            })
    
    # Get inverse dataset for each class type
    for x in count_per_class:
        limit = x["count"] / total_count * class_type_count
        temp_list_of_poses, _ = get_dataset_with_limit(
            x["class_type"], 
            get_class_type_frame_length(class_type), 
            limit)
        if class_type is "sit-up" and x["class_type"] is not "sit-up":
            list_of_poses.extend(double_array_in_array(temp_list_of_poses))
        elif class_type is not "sit-up" and x["class_type"] is "sit-up":
            list_of_poses.extend(halve_array_in_array(temp_list_of_poses))
        else:
            list_of_poses.extend(temp_list_of_poses)

        # Pop all poses and labels in list
        pop_all(temp_list_of_poses)

    # Insert gathered pose data to database
    for x in list_of_poses:
        insert_array_to_db(x, 0, "not-" + class_type)

# Initiate function
if __name__ == '__main__':
    for x in CLASS_TYPE:
        make_inverse_dataset(x)