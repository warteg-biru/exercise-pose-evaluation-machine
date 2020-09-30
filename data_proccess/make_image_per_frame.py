import os
import sys
import cv2

# get base path argument from cli
def get_source_base_dir_arg():
    try:
        path_arg = sys.argv[1]
        return path_arg
    except:
        return None

def get_output_base_dir_arg():
    try:
        path_arg = sys.argv[2]
        return path_arg
    except:
        return None

def get_last_part_of_path(path):
    last_part = os.path.basename(os.path.normpath(path))
    return last_part

def create_image(frame, output_path=''):
    cv2.imwrite(output_path, frame)

def cut_video_to_images(input_video_path, output_image_dir):
    print('input video path -> ', input_video_path)
    frame_count = 0
    folder_name = get_last_part_of_path(output_image_dir)
    stream = cv2.VideoCapture(input_video_path)
    while True:
        try:
            ret, frame = stream.read()
            frame_count += 1
            output_path = os.path.join(output_image_dir, f'{folder_name}_{str(frame_count)}.jpg')
            create_image(frame, output_path)
        except Exception as e:
            # Break at end of frame
            break


def create_output_dir(output_base_dir, exercise, output_folder_name):
    output_exercise_dir = os.path.join(output_base_dir, exercise)
    if not os.path.exists(output_exercise_dir):
        os.makedirs(output_exercise_dir)
    final_output_dir = os.path.join(output_exercise_dir, output_folder_name)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    return final_output_dir


if __name__ == '__main__':
    source_base_dir = "/home/kevin/projects/initial-pose-data/videos/to-process"
    output_base_dir = "/home/kevin/projects/initial-pose-data/images/raw"
    #  source_base_dir = get_source_base_dir_arg()
    #  output_base_dir = get_output_base_dir_arg()

    # loop through each exercise folder in base directory
    exercise_folders = os.listdir(source_base_dir)
    for exercise in exercise_folders:
        exercise_folder_dir = os.path.join(source_base_dir, exercise)
        # loop through the files in each exercise folder directory
        files = os.listdir(exercise_folder_dir)
        for file_name in files:
            input_video_path = os.path.join(exercise_folder_dir, file_name)
            # create output image dir using output base directory, exercise name, and input filename
            output_folder_name = file_name
            output_image_dir = create_output_dir(
                output_base_dir,
                exercise,
                output_folder_name
                )
            cut_video_to_images(input_video_path, output_image_dir)
