import os
import sys
import numpy as np
import math
import cv2
from keypoints_extractor import KeypointsExtractor
from kp_index import NECK, MID_HIP, L_KNEE


class AngleCalculator():
    def distance(self, x1, y1, x2, y2):
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
            
    def find_angle(self, a, b, c):
        return math.degrees(np.arccos((np.square(a) + np.square(b) - np.square(c)) / (2 * a * b)))

    def get_angle_from_three_keypoints(self, kp_1, kp_2, target_kp):
        kp_1_x = kp_1_y = kp_2_x = kp_2_y = target_kp_x = target_kp_y = None

        if type(kp_1) == dict:
            kp_1_x = kp_1['x']
            kp_1_y = kp_1['y']
            kp_2_x = kp_2['x']
            kp_2_y = kp_2['y']
            target_kp_x = target_kp['x']
            target_kp_y = target_kp['y']
        else:
            kp_1_x = kp_1[0]
            kp_1_y = kp_1[1]
            kp_2_x = kp_2[0]
            kp_2_y = kp_2[1]
            target_kp_x = target_kp[0]
            target_kp_y = target_kp[1]

        a = self.distance(kp_1_x, kp_1_y, target_kp_x, target_kp_y)
        b = self.distance(kp_2_x, kp_2_y, target_kp_x, target_kp_y)
        c = self.distance(kp_1_x, kp_1_y, kp_2_x, kp_2_y)
        return self.find_angle(a, b, c)

'''
For testing this class
'''
if __name__ == '__main__':
    sit_up_path = '/home/kevin/projects/initial-pose-data/images/sit-up/sit-up2.mp4'
    sit_up_start_point = os.path.join(sit_up_path, 'sit-up2.mp4_1.jpg')
    sit_up_mid_point = os.path.join(sit_up_path, 'sit-up2.mp4_23.jpg')

    kp_extractor = KeypointsExtractor()
    keypoints, _, _, output_image = kp_extractor.scan_image_without_normalize(sit_up_start_point)
    # print(keypoints)
    kp_neck = keypoints[NECK]
    kp_mid_hip = keypoints[MID_HIP]
    kp_left_knee = keypoints[L_KNEE]
    
    angle_calc = AngleCalculator()
    angle = angle_calc.get_angle_from_three_keypoints(kp_neck, kp_left_knee, kp_mid_hip)
    print(angle)

    cv2.imshow('image', output_image)
    cv2.waitKey(0)