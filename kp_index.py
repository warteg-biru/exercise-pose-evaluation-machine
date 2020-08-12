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

kp_index = {
    'NOSE'        : 0,
    'NECK'        : 1,
    'R_SHOULDER'  : 2,
    'R_ELBOW'     : 3,
    'R_WRIST'     : 4,
    'L_SHOULDER'  : 5,
    'L_ELBOW'     : 6,
    'L_WRIST'     : 7,
    'MID_HIP'     : 8,
    'R_HIP'       : 9,
    'R_KNEE'      : 10,
    'R_ANKLE'     : 11,
    'L_HIP'       : 12,
    'L_KNEE'      : 13,
    'L_ANKLE'     : 14,
    'R_EYE'       : 15,
    'L_EYE'       : 16,
    'R_EAR'       : 17,
    'L_EAR'       : 18,
    'L_BIG_TOE'   : 19,
    'L_SMALL_TOE' : 20,
    'L_HEEL'      : 21,
    'R_BIG_TOE'   : 22,
    'R_SMALL_TOE' : 23,
    'R_HEEL'      : 24,
}

kp_index_by_int = {
    0: 'NOSE',
    1: 'NECK',
    2: 'R_SHOULDER',
    3: 'R_ELBOW',
    4: 'R_WRIST',
    5: 'L_SHOULDER',
    6: 'L_ELBOW',
    7: 'L_WRIST',
    8: 'MID_HIP',
    9: 'R_HIP',
    10: 'R_KNEE',
    11: 'R_ANKLE',
    12: 'L_HIP',
    13: 'L_KNEE',
    14: 'L_ANKLE',
    15: 'R_EYE',
    16: 'L_EYE',
    17: 'R_EAR',
    18: 'L_EAR',
    19: 'L_BIG_TOE',
    20: 'L_SMALL_TOE',
    21: 'L_HEEL',
    22: 'R_BIG_TOE',
    23: 'R_SMALL_TOE',     
    24: 'R_HEEL',      
}

# Get keypoints number/index in keypoint list
def get_kp_index(index):
    return kp_index[index]

# Get keypoint on the lookup dictionary
def get_kp_index_by_int(index):
    return kp_index_by_int[index]