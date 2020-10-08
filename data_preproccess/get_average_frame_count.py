from moviepy.editor import VideoFileClip
import os

def get_frame_count(video_path):
    clip = VideoFileClip(video_path)
    n_frames = clip.reader.nframes
    return n_frames

'''
get_average_frame_count

@param {string} dir_path -- directory path of videos
'''
def get_average_frame_count(dir_path):
    files = os.listdir(dir_path)
    total_files = len(files)
    total_frames = 0
    for file in files:
        filepath = f'{dir_path}/{file}'
        frame_count = get_frame_count(filepath)
        total_frames += frame_count
    return total_frames // total_files

dir_path = '/home/kevin/Videos/Battlestar.Galactica.Season.3'
avg_frame_count = get_average_frame_count(dir_path)
