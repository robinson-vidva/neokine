"""MediaPipe Tasks PoseLandmarker wrapper (fully local Python backend).

Mirrors the web tool's options (docs/js/pose.js): lite/full/heavy model
variants, numPoses, and the two confidence thresholds. The .task model files
are downloaded once from Google's CDN into app/models/ and reused offline
thereafter.
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URLS = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Column layout of the (33, 5) landmark array this module returns.
X, Y, Z, VIS, PRES = 0, 1, 2, 3, 4


def ensure_model(variant):
    """Return a local path to the variant's .task, downloading it once."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "pose_landmarker_%s.task" % variant)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        os.replace(tmp, path)
    return path


def make_landmarker(variant, num_poses, min_detection, min_presence, mode="image"):
    """Create a PoseLandmarker. mode is 'image' or 'video'."""
    running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model(variant)),
        running_mode=running,
        num_poses=int(num_poses),
        min_pose_detection_confidence=float(min_detection),
        min_pose_presence_confidence=float(min_presence),
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _to_arrays(result):
    """Convert a PoseLandmarkerResult to a list of (33, 5) float arrays
    [x, y, z, visibility, presence], x/y normalized to [0, 1]. Empty if none."""
    poses = []
    for lms in result.pose_landmarks:
        poses.append(np.array(
            [[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in lms],
            dtype=float,
        ))
    return poses


def detect_image(landmarker, rgb):
    """Run pose on one RGB uint8 image. Returns list of (33, 5) arrays."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    return _to_arrays(landmarker.detect(image))


def detect_video_frame(landmarker, rgb, timestamp_ms):
    """Run pose on one RGB frame in VIDEO mode. Timestamps must be monotonic."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    return _to_arrays(landmarker.detect_for_video(image, int(timestamp_ms)))
