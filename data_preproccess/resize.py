import os
from moviepy.editor import *

# Define base path for the dataset
# base_path = '/home/kevin/Videos/dataset-pushup-squats'
base_path = '/home/kevin/Videos/dataset-pushup-squats'
input_dir = base_path + '/mp4'
output_dir = base_path + '/resized'


def transform_video(file_path, filename):
    # Resize video to optimal size
    clip = (VideoFileClip(file_path)
            .fx(vfx.resize, width=720, height=405)) # resize (keep aspect ratio)
    
    # Remove volume
    clip = clip.volumex(0)

    # Define directory and size
    # dir = 'preprocessed_videos'
    dir = output_dir
    # clip.write_videofile(dir + '/' + filename + '.mp4')
    clip.write_videofile(dir + '/' + filename)

if __name__ == '__main__':
    # Get dataset folders
    # dirs = os.listdir(base_path)
    files = os.listdir(input_dir)

    # Loop in each folder
    for idx, filename in enumerate(files):
        # file = base_path+'/'+filename
        file = input_dir + '/' + filename
        
        transform_video(file, filename)