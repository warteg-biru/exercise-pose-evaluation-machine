import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Write headers
def write_header(folder, filename):
    if not os.path.exists(f'process_results/{folder}'):
        os.mkdir(f'process_results/{folder}')
    f = open(f'process_results/{folder}/{filename}.csv', 'w')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(folder, filename, data):
    if not os.path.exists(f'process_results/{folder}'):
        os.mkdir(f'process_results/{folder}')
    f = open(f'process_results/{folder}/{filename}.csv', 'a')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
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
    if not os.path.exists(f'process_results/{folder}'):
        os.mkdir(f'process_results/{folder}')
    data = list_dict_to_dict_list(d)
    plt.plot(data["epoch"],data["loss"], label="Training Loss")
    plt.plot(data["epoch"],data["acc"], label="Training Accuracy")
    plt.plot(data["epoch"],data["val_loss"], label="Validation Loss")
    plt.plot(data["epoch"],data["val_acc"], label="Validation Accuracy")
    plt.legend(loc="upper left")
    plt.ylim(0, 1.0)
    plt.savefig(f'process_results/{folder}/{filename}.png')
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

if __name__ == "__main__":
    # Initialize logs path
    k = 10
    log_path = '/home/kevin/projects/exercise_pose_evaluation_machine/k-fold-results/training_logs/'
    
    # Get all files from folder
    files = os.listdir(log_path)
    for filename in files:
        with open(filename) as openfile:
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

            for i, x in enumerate(np.array_split(arr_body, k)):
                new_name = f'iteration-{i+1}-{name}'
                write_header(name, new_name)
                for idx, body in enumerate(x):
                    body["epoch"] = idx + 1
                    write_body(name, new_name, body)
                    x[idx] = body
                    
                plot_log_and_save(name, new_name, x)