import os
import csv

# Write headers
def write_header(filename):
    f = open(f'process_results/{filename}.csv', 'w')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(filename, data):
    f = open(f'process_results/{filename}.csv', 'a')
    with f:
        fnames = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

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
    log_path = '/home/kevin/projects/exercise_pose_evaluation_machine/k-fold-results/training_logs/'
    
    # Get all files from folder
    files = os.listdir(dataset_path)
    for filename in files:
        with open(filename) as openfile:
            name = os.path.splitext(filename)[0]
            write_header(name)
            idx = 1
            for line in openfile:
                body = {}
                if "sample" in line and "loss" in line and "acc" in line and "val_loss" in line and "val_acc" in line:
                    body["epoch"] = idx
                    body["loss"] = get_loss(line)
                    body["acc"] = get_acc(line)
                    body["val_loss"] = get_val_loss(line)
                    body["val_acc"] = get_val_acc(line)
                    write_body(name, body)
                    idx+=1