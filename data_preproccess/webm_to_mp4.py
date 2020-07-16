import os

path = '/home/kevin/Videos/dataset-pushup-squats'
output_path = '/home/kevin/Videos/dataset-pushup-squats-mp4'

for filename in os.listdir(path):
    if (filename.endswith(".webm")): #or .avi, .mpeg, whatever.
        file_path = path + '/' + filename
        output_filename = ''.join(filename.split('.')[:-1])
        output_file_path = output_path + '/' + output_filename
        command = f'ffmpeg -i {file_path} {output_file_path}.mp4'
        os.system(command)
    else:
        continue
