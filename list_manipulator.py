import math

'''
pop_all

# Pop entire list
@params {list} list to be popped
'''
# Pop all in array
def pop_all(l):
    r, l[:] = l[:], []
    return r

def get_exact_frames(frames, class_type):
    max_frames = 0
    if class_type is "sit-up":
        max_frames = 48
    else:
        max_frames = 24

    new_array = []
    interval = len(frames)/max_frames
    index = 0
    for x in range(max_frames):
        if index > max_frames-1:
            index = max_frames-1
        new_array.append(frames[math.ceil(index)])
        index+=interval

    return new_array
