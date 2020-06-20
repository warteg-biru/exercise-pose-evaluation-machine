import os
from moviepy.editor import *
from moviepy.video.fx.speedx import speedx

# Define base path for the dataset
base_path = '/mnt/c/Users/user/Videos/Bandicut/'

def cut_video(file_path, filename):
    # Get clip duration
    clip = VideoFileClip(file_path)
    duration = clip.duration
    
    # Cut video
    clip = clip.cutout(duration-4.5, duration)
    
    # Speedup clip
    clip = speedx(clip=clip, final_duration=2)

    # Define directory and size
    save_path = base_path + 'preprocessed_videos'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    # Write video
    clip.write_videofile(save_path + '/' + filename, fps=24)

if __name__ == '__main__':
    
    # Get dataset folders
    dirs = os.listdir(base_path)

    # Loop in each folder
    for idx, filename in enumerate(dirs):
        file = base_path+'/'+filename
        
        cut_video(file, filename)