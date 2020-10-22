# This will import all the widgets 
# and modules which are available in 
# tkinter and ttk module 
import cv2
import traceback
import numpy as np
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import ImageTk, Image
from pose_evaluator import pose_evaluator

sys.path.append('/home/kevin/projects/exercise_pose_evaluation_machine')
from posemachine import PoseMachine

def detect_initial_pose():
    # Creates a Tk() object 
    master = Tk() 

    # Sets the geometry of main 
    # Root window 
    master.geometry("800x627") 
    master.title("Pose Machine - Detect Initial Pose")
    panel = Label(master, text ="Detect Initial Pose")
    panel.pack(side = "top", fill = "both", expand = "yes")

    # Capture video
    cap=cv2.VideoCapture(0)

    # Open new window
    def open_new_window():
        cap.release()
        master.destroy()
        pose_evaluator()

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
        # Send to external function
        list_of_keypoints = pm.kp_extractor.get_keypoints_and_id_from_img(frame)
        try:
            for x in list_of_keypoints:
                if x["ID"] is pm.target_id:
                    # Transform keypoints list to (1, 24) matrix
                    keypoints = np.array(x['Keypoints']).flatten().astype(np.float32)
                    keypoints = np.array([keypoints])

                    # Get prediction
                    prediction = pm.init_pose_detector.predict(keypoints)
                    print(prediction)
                    pm.exercise_type = prediction
                    pm.exercise_type = "push-up" # FIXME: sementara buat ngecek lstm
                    # Open next window
                    if prediction is not -1:
                        open_new_window()
        except Exception as e:
            print("error in detect_initial_pose")
            print(traceback.format_exc())

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