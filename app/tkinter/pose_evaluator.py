# This will import all the widgets 
# and modules which are available in 
# tkinter and ttk module 
import sys
import cv2
import numpy as np
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import ImageTk, Image

sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine')
from posemachine import PoseMachine
from list_manipulator import get_exact_frames, pop_all
from detectors_keras_api.lstm_model_keras import ExerciseEvaluator



def pose_evaluator():
    # Creates a Tk() object 
    master = Tk() 

    # Sets the geometry of main 
    # Root window 
    master.geometry("800x627") 
    master.title("Pose Machine - Pose Evaluator")
    panel = Label(master, text ="Pose Evaluator")
    panel.pack(side = "top", fill = "both", expand = "yes")

    cap=cv2.VideoCapture(0)

    def open_new_window():
        cap.release()
        master.destroy()

    pm = PoseMachine.get_instance()
    exercise_evaluator = ExerciseEvaluator(pm.exercise_type)
    all_exercise_reps = []

    def video_stream():
        start = False
        end = False
        # Read frames
        # If camera not available print error and destroy
        if cap.read()[0]==False:
            cap.release()
            master.destroy()
            messagebox.showerror("Error", "Error when processing: Camera is not available!")
            exit()

        # Read frames if not exists
        _, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Process frames here
        # Send to external function
        # Get keypoint and ID data
        list_of_keypoints = pm.kp_extractor.get_keypoints_and_id_from_img(frame)

        try: 
            if list_of_keypoints == None:
                raise Exception("List of keypoints cannot be None")
            x = list_of_keypoints[0]
            if x['ID'] == pm.target_id:
                print("masuk sini gak???")
                # Transform keypoints list to array
                keypoints = np.array(x['Keypoints']).flatten()

                # Get prediction
                prediction = pm.init_pose_detector.predict(np.array([keypoints]))

                # If starting position is found and start is True then mark end
                if prediction == pm.exercise_type and start:
                    end = True
                
                # If starting position is found and end is False then mark start
                if prediction == pm.exercise_type and not end:
                    start = True

                    # If the found counter is more than one
                    # Delete frames and restart collection
                    if len(all_exercise_reps) >= 1:
                        pop_all(all_exercise_reps)

                # Add frames
                all_exercise_reps.append(pm.kp_extractor.get_keypoints_and_id_from_img_without_normalize(frame))

                # If both start and end was found 
                # send data to LSTM model and Plotter
                if start and end:
                    # Send data
                    x_low, y_low, _, _ = pm.kp_extractor.get_min_max_frames(all_exercise_reps)
                    scaler = make_min_max_scaler(all_exercise_reps, x_low, y_low)
                    normalized_reps = normalize_keypoints_from_external_scaler(all_exercise_reps, scaler)
                    reshaped_normalized_reps = [np.array(frames).flatten() for frames in normalized_reps]

                    exercise_evaluator.predict(get_exact_frames(reshaped_normalized_reps))
                    # Pop all frames in list
                    pop_all(all_exercise_reps)

                    # Restart found_counter, start flag and end flag
                    start = True
                    end = False

                    # Add frames
                    all_exercise_reps.append(keypoints)
        except Exception as e:
            print(e)

        # Convert the Image object into a TkPhoto object
        im = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(
            im.resize(
                (800, 600),
                Image.ANTIALIAS
            )
        )

        # Show image in panel
        panel.imgtk = imgtk
        panel.configure(image=imgtk)
        panel.after(10, video_stream) 

    video_stream()
    master.mainloop()