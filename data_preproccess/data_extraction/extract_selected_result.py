import os

import sys
sys.path.append("/home/binus/projects/exercise-pose-evaluation-machine")

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from list_manipulator import pop_all

# Write headers
def write_header(folder, filename):
    f = open(f'{folder}/{filename}.csv', 'w')
    with f:
        fnames = ['name', 'epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

# Write average
def write_body(folder, filename, data):
    f = open(f'{folder}/{filename}.csv', 'a')
    with f:
        fnames = ['name', 'epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writerow(data)

if __name__ == "__main__":
    from multiprocessing import Process

    # Initialize logs path
    log_path = '/home/binus/projects/exercise-pose-evaluation-machine/selected_result'
    
    # Get all files from folder
    model_type_list = os.listdir(log_path)
    for model_type in model_type_list:
        model_list = os.listdir(f'{log_path}/{model_type}')
        for model in model_list:
            folder = f'{log_path}/{model_type}/{model}'
            save_filename = f'{model}-results'
            write_header(folder, save_filename)

            experiment_list = os.listdir(folder)
            for experiment in experiment_list:
                try:
                    # If folder has .txt extension, remove it
                    fn, ext = os.path.splitext(f'{folder}/{experiment}')
                    if ext == '.txt':
                        os.rename(f'{folder}/{experiment}', f'{fn}')
                        experiment = fn.split('/')[-1]
                    
                    file_list = os.listdir(f'{folder}/{experiment}')
                    for filename in file_list:
                        if filename.startswith('average'):
                            body = {}
                            with open(f'{folder}/{experiment}/{filename}') as csv_file:
                                csv_reader = csv.reader(csv_file, delimiter=',')
                                for line_count, row in enumerate(csv_reader):
                                    if line_count > 0:
                                        body['name'] = row[0]
                                        body['epoch'] = row[1]
                                        body['loss'] = row[2]
                                        body['acc'] = row[3]
                                        body['val_loss'] = row[4]
                                        body['val_acc'] = row[5]
                            write_body(folder, save_filename, body)
                            break
                except:
                    continue