import os
from moviepy.editor import *
from moviepy.video.fx.speedx import speedx

# Define base path for the dataset
base_path = 'Videos/Saves'
save_base_path = 'Videos'
count = 0

def cut_video(folder, filename):
    # Get clip duration
    file_path = base_path + '/' + folder + '/' + filename
    clip = VideoFileClip(file_path)
    duration = clip.duration
    
    # Cut video
    clip = clip.cutout(duration-4.5, duration)
    
    # Speedup clip
    fin = 1
    if folder == 'push-up':
        fin = 1.5
    elif folder == 'sit-up':
        fin = 2
    elif folder == 'plank':
        fin = 1
    elif folder == 'dumbell-curl':
        fin = 1
    clip = speedx(clip=clip, final_duration=fin)

    # Define directory
    if not os.path.exists(save_base_path + '/preprocessed_videos'):
        os.mkdir(save_base_path + '/preprocessed_videos')
        print("Made preprocess directory!")
    if not os.path.exists(save_base_path + '/preprocessed_videos/' + folder):
        os.mkdir(save_base_path + '/preprocessed_videos/' + folder)
        print("Made " + folder + " directory!")
    save_path = save_base_path + '/preprocessed_videos/' + folder
        
    # Write video
    clip.write_videofile(save_path + '/' + folder + str(count) + '.mp4', fps=24)
    count+=1
    
    # Flip video
    clip = clip.fx(vfx.mirror_x)

    # Write flipped video
    clip.write_videofile(save_path + '/' + folder +  str(count) + '.mp4', fps=24)
    count+=1

if __name__ == '__main__':
    
    # Get dataset folders
    dirs = os.listdir(base_path)

    # Loop in each folder
    for folder in dirs:
        files = os.listdir(base_path + '/' + folder)
        count = 0
        for filename in files:
            cut_video(folder, filename)