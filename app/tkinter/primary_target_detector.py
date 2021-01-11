# This will import all the widgets 
# and modules which are available in 
# tkinter and ttk module 
import os
import sys
import cv2
import numpy as np
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import ImageTk, Image
from detect_initial_pose import detect_initial_pose

sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine')
from posemachine import PoseMachine

from PIL import Image

def primary_target_detector():
    # Creates a Tk() object 
    master = Tk() 

    # Sets the geometry of main 
    # Root window 
    master.geometry("800x627") 
    master.title("Pose Machine - Primary Target Detector")
    panel = Label(master, text ="Primary Target Detector")
    panel.pack(side = "top", fill = "both", expand = "yes")

    # Capture video
    cap=cv2.VideoCapture(0)

    # Open new window
    def open_new_window():
        cap.release()
        master.destroy()
        detect_initial_pose()

    # Initiate instance
    pm = PoseMachine.get_instance()

    def video_stream():
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
        # Get keypoint and ID data
        list_of_keypoints = pm.kp_extractor.get_upper_body_keypoints_and_id_from_img(frame)

        # Foreach keypoint predict user data
        found_counter = 0

        try: 
            for x in list_of_keypoints:
                # Transform keypoints list to (1, 16) matrix
                keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                keypoints = np.array([keypoints])
                # Get prediction
                prediction = pm.right_hand_up_detector.predict(keypoints)

                if prediction == 1:
                    found_counter+=1
                    pm.target_id = x['ID']

                if found_counter > 1:
                    print("Too many people raised their hands!")
                    break

            if found_counter == 0:
                print("No one was found")
            elif found_counter == 1:
                print("Person " + str(pm.target_id) + " raised their hand")
                open_new_window()
        except Exception as e:
            # Break at end of frame
            pass

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