"""Kinematics from a sequence of per-frame landmarks. Normalized image units.

Faithful to the web tool (docs/js/app.js): 3-frame moving-average smoothing,
path-length displacement, mean velocity = path length / elapsed time, and the
left-right asymmetry index (L - R) / (L + R) computed from total displacement.

These are visualization metrics in normalized image units per second, not a
physical measurement: no calibration, depth, or scale reference.
"""

import numpy as np


def smooth(a):
    """3-frame moving average (i-1, i, i+1), matching smoothInPlace()."""
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return a.copy()
    out = a.copy()
    for i in range(len(a)):
        lo, hi = max(0, i - 1), min(len(a) - 1, i + 1)
        out[i] = a[lo:hi + 1].mean()
    return out


def series(frames, idx, smooth_on=False):
    """(t, x, y) arrays for one joint across frames that detected a pose.

    frames: list of {"t": seconds, "poses": [ (33,5) array, ... ]}.
    Uses the first detected pose in each frame, like the web tool.
    """
    t, x, y = [], [], []
    for fr in frames:
        poses = fr.get("poses")
        if not poses:
            continue
        p = poses[0]
        t.append(fr["t"])
        x.append(p[idx, 0])
        y.append(p[idx, 1])
    t, x, y = np.array(t), np.array(x), np.array(y)
    if smooth_on and len(x) >= 2:
        x, y = smooth(x), smooth(y)
    return t, x, y


def path_length(x, y):
    """Total displacement: sum of segment lengths along the trajectory."""
    if len(x) < 2:
        return 0.0
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def mean_velocity(t, x, y):
    """Path length divided by elapsed time (normalized units / second)."""
    if len(t) < 2:
        return 0.0
    dt = t[-1] - t[0]
    return path_length(x, y) / dt if dt > 0 else 0.0


def asymmetry(frames, left_idx, right_idx, smooth_on=False):
    """(index, L, R) where index = (L - R) / (L + R) from total displacement.

    Range [-1, 1]; 0 = symmetric, positive = left moved more.
    """
    _, lx, ly = series(frames, left_idx, smooth_on)
    _, rx, ry = series(frames, right_idx, smooth_on)
    L, R = path_length(lx, ly), path_length(rx, ry)
    idx = (L - R) / (L + R) if (L + R) > 0 else 0.0
    return idx, L, R
