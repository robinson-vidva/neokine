"""IO helpers: read images, read COCO ground truth, write overlays and CSV.

Pure functions. No CLI parsing, no argument handling.
"""

import csv
import json
import os

import cv2


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(folder):
    """Return sorted absolute paths of image files directly in folder."""
    out = []
    for name in os.listdir(folder):
        if name.lower().endswith(IMAGE_EXTS):
            out.append(os.path.join(folder, name))
    return sorted(out)


def load_image(path):
    """Read an image as a BGR ndarray, or None if it cannot be read."""
    return cv2.imread(path)


def save_image(image_bgr, path):
    """Write a BGR ndarray to disk, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image_bgr)


def load_ground_truth(annotations_path):
    """Load COCO keypoint annotations.

    Returns a dict: file_name -> list of 17 (x, y, visibility) tuples,
    in COCO-17 order. Coordinates are absolute pixels as stored in the file.
    Assumes one annotation per image (true for SyRIP).
    """
    with open(annotations_path) as fh:
        data = json.load(fh)

    id_to_name = {img["id"]: img["file_name"] for img in data["images"]}
    gt = {}
    for ann in data["annotations"]:
        name = id_to_name.get(ann["image_id"])
        if name is None:
            continue
        flat = ann["keypoints"]
        pts = [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)]
        gt[name] = pts
    return gt


def open_landmarks_csv(path):
    """Open the landmarks CSV in append mode, writing the header if new.

    Returns (file_handle, csv_writer). Caller is responsible for closing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["image", "landmark_id", "x", "y", "z", "visibility"])
    return fh, writer


def write_landmark_rows(writer, image_name, landmarks):
    """Append one row per landmark. landmarks is a (33, 4) array of
    raw MediaPipe values: x, y, z, visibility."""
    for lid, lm in enumerate(landmarks):
        writer.writerow([image_name, lid, lm[0], lm[1], lm[2], lm[3]])
