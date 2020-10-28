import os

# Define base path for the dataset
base_path = "/home/kevin/Videos/exercise-raw-dataset/2020-08-21"
save_base_path = '/home/kevin/Videos/exercise-raw-dataset/2020-08-21'

def transform_video(base, folder, filename):
    # Check if dir exists
    if not os.path.exists(save_base_path + '/preprocessed_videos'):
        os.mkdir(save_base_path + '/preprocessed_videos')
        print("Made preprocess directory!")
    if not os.path.exists(save_base_path + '/preprocessed_videos/' + folder):
        os.mkdir(save_base_path + '/preprocessed_videos/' + folder)
        print("Made " + folder + " directory!")

    # Resize video to optimal size
    file_path = base + '/' + folder + '/' + filename
    save_path = save_base_path + '/preprocessed_videos/' + folder + '/' + filename + '.mp4'
    os.system("ffmpeg -i " + file_path + " -vf scale=720:404 -an " + save_path)
    print("Resized, muted and saved!")

if __name__ == '__main__':
    # Get dataset folders
    # dirs = os.listdir(base_path)
    files = os.listdir(base_path)

    # Loop in each folder
    # for folder in dirs:
    #     files = os.listdir(base_path + '/' + folder)
    for filename in files:
        transform_video(base_path, "ASD", filename)
