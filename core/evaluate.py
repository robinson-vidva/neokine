"""Validation path: compare inference against ground truth for one image.

evaluate() depends on infer() (one direction only). infer() never depends on
this module. This path only runs on annotated data.
"""

from shared.skeleton import COCO17_TO_MP33
from core.infer import infer, landmarks_to_pixels
from core.metrics import per_joint_errors, torso_length


def _map_to_coco(landmarks_px):
    """Select the 17 MediaPipe pixel points that correspond to COCO-17."""
    return [landmarks_px[mp_id] for mp_id in COCO17_TO_MP33]


def evaluate(image_bgr, gt_keypoints, pose):
    """Run inference on one image and score it against ground truth.

    gt_keypoints: list of 17 (x, y, visibility) tuples in COCO-17 order.
    Returns a per-image record: detected, torso, errors (list of 17 or None).
    """
    landmarks = infer(image_bgr, pose)
    if landmarks is None:
        return {"detected": False, "torso": None, "errors": None}

    h, w = image_bgr.shape[:2]
    px = landmarks_to_pixels(landmarks, w, h)
    pred_coco = _map_to_coco(px)

    return {
        "detected": True,
        "torso": torso_length(gt_keypoints),
        "errors": per_joint_errors(pred_coco, gt_keypoints),
    }
