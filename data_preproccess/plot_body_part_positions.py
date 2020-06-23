import numpy as np
import matplotlib.pyplot as plt

# KP ordering of body parts
NOSE        = 0
NECK        = 1
R_SHOULDER  = 2
R_ELBOW     = 3
R_WRIST     = 4
L_SHOULDER  = 5
L_ELBOW     = 6
L_WRIST     = 7
MID_HIP     = 8
R_HIP       = 9
R_KNEE      = 10
R_ANKLE     = 11
L_HIP       = 12
L_KNEE      = 13
L_ANKLE     = 14
R_EYE       = 15
L_EYE       = 16
R_EAR       = 17
L_EAR       = 18
L_BIG_TOE   = 19
L_SMALL_TOE = 20
L_HEEL      = 21
R_BIG_TOE   = 22
R_SMALL_TOE = 23
R_HEEL      = 24

body_parts_count = 25

frames_per_action = 30

# an action contains 30 frames, where each frame contains 25 keypoints
def generate_frame():
    # in this case each keypoint occupies 2 indexes (x, y number for each kp) in the array.
    # therefore we have 50 items in the kp array
    return list(np.random.rand(body_parts_count * 2))

def generate_random_action():
    return [generate_frame() for _ in range(frames_per_action)]

'''
plot_body_part_positions

@param {Integer} body_part_index -- the starting index of a body part kp in a frame
@param {List} action -- an array of frames 
@param {String} body_part_name -- The name of the body part to show as title in the plot
'''
def plot_body_part_positions(body_part_index, action, body_part_name=None):
    movements_x = [frame[body_part_index] for frame in action]
    movements_y = [frame[body_part_index + 1] for frame in action]
    frame_indexes = [idx + 1 for idx in range(len(action))]
    if body_part_name is None:
        body_part_name = 'Body part'
    # Nose x position
    plt.subplot(211)
    plt.title(f'{body_part_name} x and y Position In each frame')
    plt.ylabel(f'{body_part_name} x position')
    plt.xlabel('Frame')
    plt.plot(frame_indexes, movements_x)
    # Nose y position
    plt.subplot(212)
    plt.ylabel(f'{body_part_name} y position')
    plt.xlabel('Frame')
    plt.plot(frame_indexes, movements_y, 'r')
    plt.show()

random_action = generate_random_action()
# plot_body_part_positions(L_ELBOW, random_action, body_part_name='Left elbow')
