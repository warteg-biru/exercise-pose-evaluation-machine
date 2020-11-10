import os

import sys
sys.path.append("/home/kevin/projects/exercise_pose_evaluation_machine")

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from list_manipulator import pop_all
from data_preproccess.data_extraction.extract_k_fold_log import chunk_it, get_loss, get_acc, get_val_loss, get_val_acc, list_dict_to_dict_list

# Write headers
def write_header(folder, filename):
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine'
    if not os.path.exists(f'{base_path}/process_results/training'):
        os.mkdir(f'{base_path}/process_results/training')
    if not os.path.exists(f'{base_path}/process_results/training/{folder}'):
        os.mkdir(f'{base_path}/process_results/training/{folder}')
    f = open(f'{base_path}/process_results/training/{folder}/{filename}.csv', 'w')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(folder, filename, data):
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine'
    if not os.path.exists(f'{base_path}/process_results/training'):
        os.mkdir(f'{base_path}/process_results/training')
    if not os.path.exists(f'{base_path}/process_results/training/{folder}'):
        os.mkdir(f'{base_path}/process_results/training/{folder}')
    f = open(f'{base_path}/process_results/training/{folder}/{filename}.csv', 'a')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)
        
# Plot the data into an image
def plot_log_and_save(folder, filename, d):
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine'
    if not os.path.exists(f'{base_path}/process_results/training'):
        os.mkdir(f'{base_path}/process_results/training')
    if not os.path.exists(f'{base_path}/process_results/training/{folder}'):
        os.mkdir(f'{base_path}/process_results/training/{folder}')
    data = list_dict_to_dict_list(d)
    plt.plot(data["epoch"],data["loss"], label="Training Loss")
    plt.plot(data["epoch"],data["acc"], label="Training Accuracy")
    plt.plot(data["epoch"],data["val_loss"], label="Validation Loss")
    plt.plot(data["epoch"],data["val_acc"], label="Validation Accuracy")
    plt.legend(loc="upper left")
    plt.ylim(0, 1.0)
    plt.savefig(f'{base_path}/process_results/training/{folder}/{filename}.png')
    plt.gca().cla()

def extract_log_from_files(log_path, files):
    for filename in files:
        with open(f'{log_path}{filename}') as openfile:
            name = os.path.splitext(filename)[0]
            
            idx = 1
            arr_body = []
            for line in openfile:
                body = {}
                if "sample" in line and "loss" in line and "acc" in line and "val_loss" in line and "val_acc" in line:
                    body["epoch"] = idx
                    body["loss"] = get_loss(line)
                    body["acc"] = get_acc(line)
                    body["val_loss"] = get_val_loss(line)
                    body["val_acc"] = get_val_acc(line)
                    arr_body.append(body)

            for x in [arr_body]:
                new_name = f'training-log-{name}'
                write_header(name, new_name)
                for idx, body in enumerate(x):
                    body["epoch"] = idx + 1
                    write_body(name, new_name, body)
                    x[idx] = body
                    
                plot_log_and_save(name, new_name, x)


if __name__ == "__main__":
    from multiprocessing import Process

    # Initialize logs path
    log_path = '/home/kevin/projects/exercise_pose_evaluation_machine/models/training_logs/'
    
    # Get all files from folder
    file_list = os.listdir(log_path)

    THREADS = []

    for files in chunk_it(file_list, 5):
        thread = Process(target=extract_log_from_files, args=(log_path, files))
        thread.start()
        THREADS.append(thread)
    for t in THREADS:
        t.join()
    pop_all(THREADS)