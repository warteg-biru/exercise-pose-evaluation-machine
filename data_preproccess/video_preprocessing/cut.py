import os
from moviepy.editor import *
from moviepy.video.fx.speedx import speedx

# Define base path for the dataset
base_path = 'Videos/Saves'
save_base_path = 'Videos'

def cut_video(folder, filename, count):
    # Get clip duration
    file_path = base_path + '/' + folder + '/' + filename
    clip = VideoFileClip(file_path)
    duration = clip.duration
    
    # Cut video
    clip = clip.cutout(duration-5, duration)
    temp_clip = clip
    temp_count = count
    
    # Save processed clip
    # Define directory
    if not os.path.exists(save_base_path + '/Cut'):
        os.mkdir(save_base_path + '/Cut')
        print("Made Cut directory!")
    if not os.path.exists(save_base_path + '/Cut/' + folder):
        os.mkdir(save_base_path + '/Cut/' + folder)
        print("Made " + folder + " directory!")
    save_path = save_base_path + '/Cut/' + folder
        
    # Write video
    temp_clip.write_videofile(save_path + '/' + folder + str(temp_count) + '.mp4')
    temp_count+=1
    
    # Flip video
    temp_clip = temp_clip.fx(vfx.mirror_x)

    # Write flipped video
    temp_clip.write_videofile(save_path + '/' + folder +  str(temp_count) + '.mp4')
    temp_count+=1
    
    # Save optimized clip
    # Speedup clip
    fin = 1
    if folder == 'push-up':
        fin = 1
    elif folder == 'plank':
        fin = 1
    elif folder == 'dumbell-curl':
        fin = 1
    else:
        fin = 2
    clip = speedx(clip=clip, final_duration=fin)

    # Define directory
    if not os.path.exists(save_base_path + '/Optimized'):
        os.mkdir(save_base_path + '/Optimized')
        print("Made Optimized directory!")
    if not os.path.exists(save_base_path + '/Optimized/' + folder):
        os.mkdir(save_base_path + '/Optimized/' + folder)
        print("Made " + folder + " directory!")
    save_path = save_base_path + '/Optimized/' + folder
        
    # Write video
    clip.write_videofile(save_path + '/' + folder + str(count) + '.mp4', fps=24)
    count+=1
    
    # Flip video
    clip = clip.fx(vfx.mirror_x)

    # Write flipped video
    clip.write_videofile(save_path + '/' + folder +  str(count) + '.mp4', fps=24)
    count+=1

    return count

if __name__ == '__main__':
    
    # Get dataset folders
    dirs = os.listdir(base_path)

    # Loop in each folder
    for folder in dirs:
        files = os.listdir(base_path + '/' + folder)
        count = 0
        for filename in files:
            count = cut_video(folder, filename, count)