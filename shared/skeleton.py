"""Static constants: MediaPipe 33-landmark skeleton and the COCO-17 mapping.

Pure data only. Consumed by core/. No logic, no dependencies.
"""

# MediaPipe Pose (legacy mp.solutions.pose) landmark names, indexed 0..32.
MEDIAPIPE_LANDMARK_NAMES = [
    "nose",              # 0
    "left_eye_inner",    # 1
    "left_eye",          # 2
    "left_eye_outer",    # 3
    "right_eye_inner",   # 4
    "right_eye",         # 5
    "right_eye_outer",   # 6
    "left_ear",          # 7
    "right_ear",         # 8
    "mouth_left",        # 9
    "mouth_right",       # 10
    "left_shoulder",     # 11
    "right_shoulder",    # 12
    "left_elbow",        # 13
    "right_elbow",       # 14
    "left_wrist",        # 15
    "right_wrist",       # 16
    "left_pinky",        # 17
    "right_pinky",       # 18
    "left_index",        # 19
    "right_index",       # 20
    "left_thumb",        # 21
    "right_thumb",       # 22
    "left_hip",          # 23
    "right_hip",         # 24
    "left_knee",         # 25
    "right_knee",        # 26
    "left_ankle",        # 27
    "right_ankle",       # 28
    "left_heel",         # 29
    "right_heel",        # 30
    "left_foot_index",   # 31
    "right_foot_index",  # 32
]

# Skeleton edges between MediaPipe landmark ids (for overlay drawing).
MEDIAPIPE_EDGES = [
    # face
    (0, 2), (2, 7), (0, 5), (5, 8), (9, 10),
    # arms
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # legs
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Ground-truth keypoint order (COCO-17), as stored in the SyRIP annotations.
COCO17_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# For each COCO-17 index, the MediaPipe-33 landmark id it maps to.
# GT eye -> MediaPipe center eye (left_eye=2, right_eye=5).
COCO17_TO_MP33 = [
    0,   # nose          -> nose
    2,   # left_eye      -> left_eye (center)
    5,   # right_eye     -> right_eye (center)
    7,   # left_ear      -> left_ear
    8,   # right_ear     -> right_ear
    11,  # left_shoulder -> left_shoulder
    12,  # right_shoulder-> right_shoulder
    13,  # left_elbow    -> left_elbow
    14,  # right_elbow   -> right_elbow
    15,  # left_wrist    -> left_wrist
    16,  # right_wrist   -> right_wrist
    23,  # left_hip      -> left_hip
    24,  # right_hip     -> right_hip
    25,  # left_knee     -> left_knee
    26,  # right_knee    -> right_knee
    27,  # left_ankle    -> left_ankle
    28,  # right_ankle   -> right_ankle
]

# COCO-17 index groups used to break out metrics.
COCO17_FACE_IDX = [0, 1, 2, 3, 4]                 # nose, eyes, ears
COCO17_LIMB_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # shoulders..ankles

# COCO-17 indices used for the torso reference (shoulder-hip distance).
TORSO_LEFT = (5, 11)   # left_shoulder, left_hip
TORSO_RIGHT = (6, 12)  # right_shoulder, right_hip
