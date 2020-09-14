import os
from moviepy.editor import *
from moviepy.video.fx.speedx import speedx

# Define base path for the dataset
base_path = '/home/kevin/projects/auto-cut/edited'
save_path = '/home/kevin/projects/auto-cut/sit-up-optimized'

final_duration = {
    'plank': 1,
    'situp': 2,
    'pushup': 1,
    'dumbell-curl': 1,
    'squat': 2,
}

def cut_video(file_path, filename):
    # Get clip duration
    clip = VideoFileClip(file_path)
    duration = clip.duration
    
    # Speedup clip
    clip = speedx(clip=clip, final_duration=final_duration['situp'])

    # Define directory and size
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
