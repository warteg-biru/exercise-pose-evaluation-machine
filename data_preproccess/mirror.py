from moviepy.editor import VideoFileClip, vfx
import os

path = '/home/kevin/Videos/dataset-pushup-squats/cut/pushup'
output_path = '/home/kevin/Videos/dataset-pushup-squats/cut-mirrored/pushup'

def main():
    for filename in os.listdir(path):
        file_path = path + '/' + filename
        output_file_path = output_path + '/' + filename

        video = VideoFileClip(file_path)
        out = video.fx(vfx.mirror_x)
        out.write_videofile(output_file_path)

if __name__ == '__main__':
    main()
