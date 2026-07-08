"""Validation metrics: torso reference, MPJPE, PCK, and aggregation.

Operates only on aligned COCO-17 arrays. No inference, no IO.
"""

import math

from shared.skeleton import (
    COCO17_FACE_IDX,
    COCO17_LIMB_IDX,
    TORSO_LEFT,
    TORSO_RIGHT,
)

PCK_ALPHA = 0.2
VIS_LABELED = 2  # only score against clean (visibility == 2) GT joints


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def torso_length(gt):
    """Mean of left and right shoulder-hip distances, using only sides whose
    shoulder and hip are both visibility == 2. Returns None if neither side
    is usable. gt is a list of 17 (x, y, visibility) tuples."""
    refs = []
    for sh, hp in (TORSO_LEFT, TORSO_RIGHT):
        if gt[sh][2] == VIS_LABELED and gt[hp][2] == VIS_LABELED:
            refs.append(_dist(gt[sh], gt[hp]))
    if not refs:
        return None
    return sum(refs) / len(refs)


def per_joint_errors(pred_px, gt):
    """Per-joint pixel error for the 17 COCO joints.

    pred_px: (17, 2) predicted pixel coords (MediaPipe mapped to COCO order).
    gt: list of 17 (x, y, visibility) tuples.
    Returns a list of 17 entries: pixel error for joints with visibility == 2,
    otherwise None.
    """
    out = []
    for i in range(17):
        if gt[i][2] == VIS_LABELED:
            out.append(_dist((pred_px[i][0], pred_px[i][1]), gt[i]))
        else:
            out.append(None)
    return out


def _summarize(errors, indices, torso):
    """Given per-joint errors and the joint indices to include, return
    (sum_error, n_error, n_correct_pck, n_pck) restricted to those indices.
    PCK counts require a torso reference; if torso is None, PCK is not counted.
    """
    sum_err = 0.0
    n_err = 0
    n_correct = 0
    n_pck = 0
    thresh = None if torso is None else PCK_ALPHA * torso
    for i in indices:
        e = errors[i]
        if e is None:
            continue
        sum_err += e
        n_err += 1
        if thresh is not None:
            n_pck += 1
            if e <= thresh:
                n_correct += 1
    return sum_err, n_err, n_correct, n_pck


def aggregate(records):
    """Aggregate per-image evaluation records into overall / limb / face
    metrics. Each record is a dict with keys:
        detected (bool), torso (float or None), errors (list of 17 or None).

    Returns a dict with counts and, for each group, mean MPJPE (pixels) and
    mean PCK@0.2 (torso-normalized).
    """
    groups = {
        "overall": list(range(17)),
        "limb": COCO17_LIMB_IDX,
        "face": COCO17_FACE_IDX,
    }
    acc = {g: [0.0, 0, 0, 0] for g in groups}  # sum_err, n_err, n_correct, n_pck

    images_total = len(records)
    images_detected = 0
    images_pck_skipped = 0  # detected but no usable torso reference

    for rec in records:
        if not rec["detected"]:
            continue
        images_detected += 1
        if rec["torso"] is None:
            images_pck_skipped += 1
        for g, idx in groups.items():
            s, n, c, p = _summarize(rec["errors"], idx, rec["torso"])
            acc[g][0] += s
            acc[g][1] += n
            acc[g][2] += c
            acc[g][3] += p

    result = {
        "images_total": images_total,
        "images_detected": images_detected,
        "detection_rate": (images_detected / images_total) if images_total else 0.0,
        "images_pck_skipped": images_pck_skipped,
        "groups": {},
    }
    for g in groups:
        sum_err, n_err, n_correct, n_pck = acc[g]
        result["groups"][g] = {
            "mpjpe_px": (sum_err / n_err) if n_err else None,
            "n_joints_mpjpe": n_err,
            "pck": (n_correct / n_pck) if n_pck else None,
            "n_joints_pck": n_pck,
        }
    return result
