from db_entity import get_count, get_dataset_with_limit, insert_array_to_db
from keypoints_extractor import pop_all

CLASS_TYPE = [
    "dumbell-curl",
    "push-up",
    "sit-up",
    # "squat",
    "plank"
]

def make_inverse_dataset(class_type):
    total_count = 0
    count_per_class = []
    class_type_count = get_count(class_type)
    list_of_poses = []
    list_of_labels = []

    for x in CLASS_TYPE:
        if x != class_type:
            curr_count = get_count(x)
            total_count+=curr_count
            count_per_class.append({
                'count': curr_count,
                'class_type': x
            })
    
    for x in count_per_class:
        limit = x["count"] / total_count * class_type_count
        temp_list_of_poses, temp_list_of_labels = get_dataset_with_limit(x["class_type"], limit)
        list_of_poses.extend(temp_list_of_poses)
        list_of_labels.extend(temp_list_of_labels)

        pop_all(temp_list_of_poses)
        pop_all(temp_list_of_labels)

    for x in list_of_poses:
        insert_array_to_db(x, 0, "not-" + class_type)

if __name__ == '__main__':
    for x in CLASS_TYPE:
        make_inverse_dataset(x)