"""Pose inference. Input is an RGB image; no ground truth required.

infer() never touches ground truth and never calls evaluate(). This is the
path the future app/web tool reuses unchanged.
"""

import cv2
import numpy as np

from shared.skeleton import MEDIAPIPE_EDGES


def create_pose(model_complexity=2):
    """Create a MediaPipe legacy Pose object for static images.

    static_image_mode=True: each image is an independent still, not a video
    frame. model_complexity=2: best-available (heavy) model; this is offline
    batch work so speed is irrelevant. Confidence thresholds are left at the
    library defaults (0.5 / 0.5) and are not tuned.

    Imported lazily so that modules importing this file (e.g. metrics) do not
    require mediapipe to be installed.
    """
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
    )


def infer(image_bgr, pose):
    """Run MediaPipe Pose on one BGR image.

    Returns a (33, 4) float array of raw MediaPipe landmarks
    (x, y, z, visibility), where x/y are normalized to [0, 1].
    Returns None if no pose is detected.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks is None:
        return None
    lms = result.pose_landmarks.landmark
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms], dtype=float
    )


def landmarks_to_pixels(landmarks, width, height):
    """Convert raw normalized (33, 4) landmarks to a (33, 2) pixel array."""
    px = np.empty((len(landmarks), 2), dtype=float)
    px[:, 0] = landmarks[:, 0] * width
    px[:, 1] = landmarks[:, 1] * height
    return px


def draw_overlay(image_bgr, landmarks):
    """Draw the MediaPipe skeleton on a copy of image_bgr and return it."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    px = landmarks_to_pixels(landmarks, w, h)
    for a, b in MEDIAPIPE_EDGES:
        pa = (int(round(px[a, 0])), int(round(px[a, 1])))
        pb = (int(round(px[b, 0])), int(round(px[b, 1])))
        cv2.line(out, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
    for i in range(len(px)):
        c = (int(round(px[i, 0])), int(round(px[i, 1])))
        cv2.circle(out, c, 3, (0, 0, 255), -1, cv2.LINE_AA)
    return out
