import os
import sys
import cv2
import traceback
import numpy as np
from datetime import datetime

sys.path.append("/home/kevin/projects/exercise_pose_evaluation_machine")
from detectors_keras_api.initial_pose_detector_keras import InitialPoseDetector
from detectors_keras_api.situp_init_pose_detector import SitUpInitialPoseDetector
from keypoints_extractor import KeypointsExtractor

class RepCapture():
    def __init__(self, exercise_name):
        self.frames = []
        self.frame_count = 0
        self.is_capturing = False
        self.begin_frames_offset = 4
        self.end_frames_offset = 4
        self.end_frames_offset_count = 0
        if exercise_name == "sit-up":
            self.min_frame_count = 30
            self.max_frame_count = 120
        elif exercise_name == "push-up":
            self.min_frame_count = 15
            self.max_frame_count = 120

    def is_initial_pose(self, prediction):
        if prediction is None:
            return False
        return prediction > 0.975
    
    def is_correct_frame_count(self):
        return self.frame_count > (self.min_frame_count + self.begin_frames_offset)\
            and self.frame_count < (self.max_frame_count + self.end_frames_offset)

    def capture(self, prediction, frame):
        # Prepare to capture frames
        if self.is_initial_pose(prediction) and self.is_capturing is False:
            self.is_capturing = True
            self.frame_count = 0

        # Capture Frames
        if self.is_capturing:
            self.frames.append(frame)
            self.frame_count += 1

        # capture frames as many as ending_frames offset, and then export frames
        if self.is_initial_pose(prediction)\
                and self.is_capturing\
                and self.is_correct_frame_count():
            if self.end_frames_offset_count < self.end_frames_offset:
                self.frames.append(frame)
                self.frame_count += 1
                self.end_frames_offset_count += 1
            else:
                rep_frames = self.export_frames()
                self.reset_frames()
                self.is_capturing = False
                return rep_frames

        # Recover from not capturing
        if self.frame_count > self.max_frame_count and self.is_capturing:
            self.is_capturing = False
            self.reset_frames()

    def export_frames(self):
        print("Export frames now")
        print("Num of frames ==", self.frame_count)
        return self.frames[self.begin_frames_offset:]

    def reset_frames(self):
        self.frames = []
        self.frame_count = 0
        self.end_frames_offset_count = 0

    def reset(self):
        self.reset_frames()
        self.is_capturing = False
    


class AutomaticCutter():
    def __init__(self, input_dir, output_dir, exercise_name):
        self.kp_extractor = KeypointsExtractor()
        self.init_pose_detector = SitUpInitialPoseDetector()
        self.rep_cap = RepCapture(exercise_name)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_fps = 24

    '''
    predict initial pose
    '''
    def predict_init_pose(self, imageToProcess):
        list_of_keypoints = self.kp_extractor.get_keypoints_and_id_from_img(imageToProcess)
        try: 
            for x in list_of_keypoints:
                keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                keypoints = np.array([keypoints])
                # Get prediction
                prediction = self.init_pose_detector.predict(keypoints)
                print("Initital pose prediction result: ", prediction)
                return prediction
        except Exception as e:
            # Break at end of frame
            traceback.print_exc()
            print(e)
            pass

    '''
    flip frames horizontally
    '''
    def flip_frames_horizontal(self, frames):
        flipped_frames = [cv2.flip(frame, 1) for frame in frames]
        return flipped_frames

    '''
    write video to mp4 file
    '''
    def write_video(self, frames, width, height):
        file_name = str(datetime.now()) + ".mp4"
        file_path = os.path.join(self.output_dir, file_name)
        vid_writer = cv2.VideoWriter(
            file_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.output_fps,
            (int(width), int(height)),
        )
        for frame in frames:
            vid_writer.write(frame)
        vid_writer.release()
        
    '''
    export frames to a video file
    '''
    def output(self, frames, width, height):
        self.write_video(frames, width, height)
        flipped_frames = self.flip_frames_horizontal(frames)
        self.write_video(flipped_frames, width, height)

    '''
    process single video
    '''
    def process_video(self, file_path):
        stream = cv2.VideoCapture(file_path)
        # Get video properties
        fps = stream.get(cv2.CAP_PROP_FPS)
        width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Start stream
        has_frame, frame = stream.read()
        while has_frame:
            cv2.imshow("vid", frame)
            prediction = self.predict_init_pose(frame)
            rep_frames = self.rep_cap.capture(prediction, frame)
            if (rep_frames is not None):
                self.output(rep_frames, width, height)
            # key = cv2.waitKey(int(1000/fps)) # ini untuk play video in "real time"
            key = cv2.waitKey(1)
            # Quit
            if key == ord('q'):
                break
            has_frame, frame = stream.read()
        stream.release()

    '''
    process a folder of videos
    '''
    def process_folder(self):
        files = os.listdir(self.input_dir)
        print(files)
        for file_name in files:
            file_path = os.path.join(self.input_dir, file_name)
            self.rep_cap.reset()
            self.process_video(file_path)


'''
Get input dir and output dir from cli
'''
def get_args():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    return input_dir, output_dir


if __name__ == "__main__":
    # video_path = "/home/kevin/projects/test_auto_cut_videos/1.mp4"
    # video_path = "/home/kevin/projects/sit-up1.mp4"
    input_dir = "/home/kevin/projects/auto-cut/sit-up"
    output_dir = "/home/kevin/projects/auto-cut/sit-up-cut"

    auto_cut = AutomaticCutter(input_dir=input_dir, output_dir=output_dir, exercise_name= "sit-up")
    auto_cut.process_folder()
