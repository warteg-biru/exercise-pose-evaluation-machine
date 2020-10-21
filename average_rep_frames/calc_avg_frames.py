import os
import cv2
from tqdm import tqdm
import csv

def count_frames(video):
    # initialize the total number of frames read
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()
        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    return total


def process_exercise_dir(dir):
    total_frame_counts = []
    total_fps = []
    total_duration = []
    for file in tqdm(os.listdir(dir), desc=f"processing - {dir}"):
        cap = cv2.VideoCapture(os.path.join(dir, file))

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_fps.append(fps)

        file_frame_count = count_frames(cap)
        total_frame_counts.append(file_frame_count)

        duration = file_frame_count / fps
        total_duration.append(duration)

    result = {
        "exercise_name": dir,
        "avg_frame_count": sum(total_frame_counts) / len(total_frame_counts),
        "avg_fps": sum(total_fps) / len(total_fps),
        "avg_duration": sum(total_duration) / len(total_duration)
    }

    return result


def write_to_csv(results):
    f = open('avg_frame_count.csv', 'w')
    with f:
        # Header
        fnames = [x["exercise_name"] for x in results]
        fnames.insert(0, "metric")

        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()
        # row
        row = {}
        row["metric"] = "avg_frame_count"
        for x in results:
            row[x["exercise_name"]] = x["avg_frame_count"]
        writer.writerow(row)

        row = {}
        row["metric"] = "avg_fps"
        for x in results:
            row[x["exercise_name"]] = x["avg_fps"]
        writer.writerow(row)

        row = {}
        row["metric"] = "avg_duration"
        for x in results:
            row[x["exercise_name"]] = x["avg_duration"]
        writer.writerow(row)
        

def main():
    SQUAT_DIR = "squat"
    PUSH_UP_DIR = "push_up"
    SIT_UP_DIR = "sit_up"

    exercise_dirs = [SQUAT_DIR, PUSH_UP_DIR, SIT_UP_DIR]

    results = []
    for dir in exercise_dirs:
        result = process_exercise_dir(dir)
        results.append(result)

    write_to_csv(results)


if __name__ == "__main__":
    main()
