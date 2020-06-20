import os
from moviepy.editor import *

# Define base path for the dataset
base_path = '/home/kevin/projects/dataset_original'

def transform_video(file_path, filename):
    # Resize video to optimal size
    clip = (VideoFileClip(file_path)
            .fx(vfx.resize, width=405, height=720)) # resize (keep aspect ratio)
    
    # Remove volume
    clip = clip.volumex(0)

    # Define directory and size
    dir = 'preprocessed_videos'
    clip.write_videofile(dir + '/' + filename + '.mp4')

if __name__ == '__main__':
    # Get dataset folders
    dirs = os.listdir(base_path)

    # Loop in each folder
    for idx, filename in enumerate(dirs):
        file = base_path+'/'+filename
        
        transform_video(file, filename)