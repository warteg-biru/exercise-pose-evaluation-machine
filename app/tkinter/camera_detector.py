import tkinter as tk
from tkinter.ttk import *
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox
from primary_target_detector import primary_target_detector

def camera_detector():
    width, height = 800, 627
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    root = tk.Tk()
    root.title("Pose Machine - Camera Detector")
    root.bind('<Escape>', lambda e: root.quit())
    lmain = tk.Label(root)
    lmain.pack()

    def check_camera():
        if cap.read()[0]==False:
            MsgBox = tk.messagebox.askquestion('No Camera Available','Do you want to retry?',icon = 'warning')
            if MsgBox == 'yes':
                root.destroy()
                camera_detector()
            else:
                root.destroy()
        else:
            cap.release()
            root.destroy()
            primary_target_detector()

    check_camera()
    root.mainloop()

camera_detector()