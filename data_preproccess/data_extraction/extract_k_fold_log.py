import os

import sys
sys.path.append("/home/kevin/projects/exercise_pose_evaluation_machine")

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from list_manipulator import pop_all

# Write headers
def write_header(folder, filename):
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine'
    if not os.path.exists(f'{base_path}/process_results/k_fold'):
        os.mkdir(f'{base_path}/process_results/k_fold')
    if not os.path.exists(f'{base_path}/process_results/k_fold/{folder}'):
        os.mkdir(f'{base_path}/process_results/k_fold/{folder}')
    f = open(f'{base_path}/process_results/k_fold/{folder}/{filename}.csv', 'w')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(folder, filename, data):
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine'
    if not os.path.exists(f'{base_path}/process_results/k_fold'):
        os.mkdir(f'{base_path}/process_results/k_fold')
    if not os.path.exists(f'{base_path}/process_results/k_fold/{folder}'):
        os.mkdir(f'{base_path}/process_results/k_fold/{folder}')
    f = open(f'{base_path}/process_results/k_fold/{folder}/{filename}.csv', 'a')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

# Write average
def write_avg_body(folder, filename, data):
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine'
    if not os.path.exists(f'{base_path}/process_results/k_fold'):
        os.mkdir(f'{base_path}/process_results/k_fold')
    if not os.path.exists(f'{base_path}/process_results/k_fold/{folder}'):
        os.mkdir(f'{base_path}/process_results/k_fold/{folder}')
    f = open(f'{base_path}/process_results/k_fold/{folder}/{filename}.csv', 'w')
    with f:
        fnames = ['name', 'epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        writer.writerow(data)

# Convert from list of dictionaries to dictionaries of list
# Type conversion made specifically for this case
def list_dict_to_dict_list(data):
    body = {}
    body["epoch"] = [d["epoch"] for d in data]
    body["loss"] = [float(d["loss"]) for d in data]
    body["acc"] = [float(d["acc"]) for d in data]
    body["val_loss"] = [float(d["val_loss"]) for d in data]
    body["val_acc"] = [float(d["val_acc"]) for d in data]
    return body
        
# Plot the data into an image
def plot_log_and_save(folder, filename, d):
    base_path = '/home/kevin/projects/exercise_pose_evaluation_machine'
    if not os.path.exists(f'{base_path}/process_results/k_fold'):
        os.mkdir(f'{base_path}/process_results/k_fold')
    if not os.path.exists(f'{base_path}/process_results/k_fold/{folder}'):
        os.mkdir(f'{base_path}/process_results/k_fold/{folder}')
    data = list_dict_to_dict_list(d)
    plt.plot(data["epoch"],data["loss"], label="Training Loss")
    plt.plot(data["epoch"],data["acc"], label="Training Accuracy")
    plt.plot(data["epoch"],data["val_loss"], label="Validation Loss")
    plt.plot(data["epoch"],data["val_acc"], label="Validation Accuracy")
    plt.legend(loc="upper left")
    plt.ylim(0, 1.0)
    plt.savefig(f'{base_path}/process_results/k_fold/{folder}/{filename}.png')
    plt.gca().cla()

# Get log data
def get_loss(line):
    return line[line.find('loss') + len("loss: "): line.find(" - acc")]

def get_acc(line):
    return line[line.find('acc') + len("acc: "): line.find(" - val_loss")]

def get_val_loss(line):
    return line[line.find('val_loss') + len("val_loss: "): line.find(" - val_acc")]

def get_val_acc(line):
    return line[line.find('val_acc') + len("val_acc: "): len(line)-1]

def extract_log_from_files(k, log_path, files):
    for filename in files:
        with open(f'{log_path}{filename}') as openfile:
            name = filename.replace(':', '.')
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

            try:
                if len(arr_body) < 1:
                    raise Exception("Log cannot be empty")

                final_result = []
                for i, x in enumerate(np.array_split(arr_body, k)):
                    new_name = f'iteration-{i+1}-{name}'
                    write_header(name, new_name)
                    last_row = len(x)
                    for idx, body in enumerate(x):
                        body["epoch"] = idx + 1
                        write_body(name, new_name, body)
                        x[idx] = body
                        if last_row == idx + 1:
                            final_result.append(body)
                    plot_log_and_save(name, new_name, x)

                new_body = {}
                new_body["name"] = name
                new_body["epoch"] = final_result[0]["epoch"]
                new_body["loss"] = sum([float(body["loss"]) for body in final_result]) / k
                new_body["acc"] = sum([float(body["acc"]) for body in final_result]) / k
                new_body["val_loss"] = sum([float(body["val_loss"]) for body in final_result]) / k
                new_body["val_acc"] = sum([float(body["val_acc"]) for body in final_result]) / k
                new_name = f'average-{name}'
                write_avg_body(name, new_name, new_body)
            except Exception as ex:
                print(f'Cannot process log {name}: {str(ex)}')

def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


if __name__ == "__main__":
    from multiprocessing import Process

    # Initialize logs path
    k = 10
    log_path = '/home/kevin/projects/exercise_pose_evaluation_machine/k_fold_results/training_logs/'
    
    # Get all files from folder
    file_list = os.listdir(log_path)

    THREADS = []

    for files in chunk_it(file_list, 3):
        thread = Process(target=extract_log_from_files, args=(k, log_path, files))
        thread.start()
        THREADS.append(thread)
    for t in THREADS:
        t.join()
    pop_all(THREADS)