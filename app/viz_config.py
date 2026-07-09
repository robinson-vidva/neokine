"""Overlay styling for the Streamlit app: region colors, skeleton connections,
and display-filter groups. Ported from the web tool (docs/js/pose.js) so the
Python app and the web app draw the same skeleton the same way.

Pure data. The landmark names/edges themselves live in shared/skeleton.py.
"""

# Region colors. Vivid, high-saturation so they read against skin and light
# backgrounds; combined with the dark outline in overlay.py for contrast.
REGION_COLORS = {
    "left": "#00e676",   # bright green
    "right": "#ff9100",  # bright orange
    "torso": "#2979ff",  # bright blue
    "face": "#ffffff",   # white (neutral, visible via the dark outline; not "scary" red)
}

# Connections grouped by region so each is colored distinctly. A connection is
# drawn only when both endpoints are currently shown (display filter).
CONNECTIONS = {
    "face": [(0, 2), (2, 7), (0, 5), (5, 8), (9, 10)],
    "torso": [(11, 12), (11, 23), (12, 24), (23, 24)],
    "left": [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
             (23, 25), (25, 27), (27, 29), (29, 31), (27, 31)],
    "right": [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
              (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)],
}

# Display-filter groups. `members` = every landmark in the group; `mains` = the
# joints that get a text label. Face is off by default (its connections read as
# a distracting mask, not body pose).
GROUPS = {
    "face":  {"label": "Face",        "members": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "mains": [0, 2, 5, 7, 8], "on": True},
    "torso": {"label": "Torso",       "members": [11, 12, 23, 24],                   "mains": [11, 12, 23, 24], "on": True},
    "left":  {"label": "Left limbs",  "members": [13, 15, 25, 27],                   "mains": [13, 15, 25, 27], "on": True},
    "right": {"label": "Right limbs", "members": [14, 16, 26, 28],                   "mains": [14, 16, 26, 28], "on": True},
    "hands": {"label": "Hands",       "members": [17, 18, 19, 20, 21, 22],           "mains": [], "on": True},
    "feet":  {"label": "Feet",        "members": [29, 30, 31, 32],                   "mains": [], "on": True},
}

# index -> group key
GROUP_OF = [None] * 33
for _key, _g in GROUPS.items():
    for _i in _g["members"]:
        GROUP_OF[_i] = _key

# Joints drawn larger.
MAJOR = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

# Joints eligible for a text label (the main joints across the labeled groups).
LABELABLE = sorted(set(
    GROUPS["face"]["mains"] + GROUPS["torso"]["mains"]
    + GROUPS["left"]["mains"] + GROUPS["right"]["mains"]
))

# Left/right joint pairs offered for the asymmetry metric (matches the web tool).
ASYM_PAIRS = {
    "wrists": (15, 16),
    "elbows": (13, 14),
    "shoulders": (11, 12),
    "hips": (23, 24),
    "knees": (25, 26),
    "ankles": (27, 28),
}


def joint_color(i):
    """Face gray; left limbs green (odd indices), right limbs amber (even)."""
    if i <= 10:
        return REGION_COLORS["face"]
    return REGION_COLORS["left"] if i % 2 == 1 else REGION_COLORS["right"]
