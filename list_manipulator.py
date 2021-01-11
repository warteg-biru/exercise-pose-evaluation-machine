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
    if class_type == "sit-up" or class_type == "squat":
        max_frames = 48
    else:
        max_frames = 24

    new_array = []
    frame_count = len(frames)
    difference = max_frames - frame_count
    comparison = max_frames/frame_count
    if frame_count < max_frames:
        extra = int(math.floor(comparison)) if comparison < 1.3 else int(math.ceil(comparison))
        last_keypoint = frames[-1]
        if extra == 1:
            new_array.extend(frames)
            for repeat in range(difference):
                new_array.append(last_keypoint)
        else:
            for index in range(frame_count):
                for repeat in range(extra):
                    if len(new_array) < max_frames:
                        new_array.append(frames[index])
                    else:
                        break
    else:
        interval = frame_count/max_frames
        index = 0
        for x in range(max_frames):
            if index > max_frames-1:
                index = max_frames-1
            new_array.append(frames[math.ceil(index)])
            index+=interval

    return new_array

def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]
