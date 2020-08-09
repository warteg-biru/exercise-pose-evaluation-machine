import os
import sys

# Define base path for the dataset
base_path = "/home/kevin/projects/dataset-handsup-to-exercise"
save_base_path = '/home/kevin/projects/dataset-handsup-to-exercise'

# get base path argument from cli
def get_base_path_argument():
    try:
        path_arg = sys.argv[1]
        return path_arg
    except:
        return None

def transform_video(base, filename):
    # Check if dir exists
    if not os.path.exists(save_base_path + '/preprocessed_videos'):
        os.mkdir(save_base_path + '/preprocessed_videos')
        print("Made preprocess directory!")

    # Resize video to optimal size
    file_path = base + '/' + filename
    save_path = save_base_path + '/preprocessed_videos/' + filename + '.mp4'
    os.system("ffmpeg -i " + file_path + " -vf scale=720:404 -an " + save_path)
    print("Resized, muted and saved!")

if __name__ == '__main__':
    base_path_arg = get_base_path_argument()
    if base_path_arg is not None:
        base_path = base_path_arg
        save_base_path = base_path_arg

    # Get dataset folders
    # dirs = os.listdir(base_path)
    files = os.listdir(base_path)

    # Loop in each folder
    # for folder in dirs:
    #     files = os.listdir(base_path + '/' + folder)
    for filename in files:
        transform_video(base_path, filename)
