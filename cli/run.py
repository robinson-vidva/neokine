"""Thin CLI over core/. Inference by default; --validate adds metrics.

Parses args, calls core, prints results. No pose or metric logic lives here.
"""

import argparse
import os
import sys

# Allow running as `python cli/run.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import io
from core.infer import create_pose, infer, draw_overlay
from core.evaluate import evaluate
from core.metrics import aggregate


def _overlay_path(output_dir, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, "overlays", stem + "_overlay.jpg")


def run_inference(image_paths, output_dir, pose):
    """Inference mode: overlay + landmarks CSV. No ground truth."""
    csv_path = os.path.join(output_dir, "landmarks.csv")
    fh, writer = io.open_landmarks_csv(csv_path)
    processed = 0
    detected = 0
    try:
        for path in image_paths:
            image = io.load_image(path)
            if image is None:
                print("skip (unreadable): " + path)
                continue
            processed += 1
            name = os.path.basename(path)
            landmarks = infer(image, pose)
            if landmarks is None:
                continue
            detected += 1
            io.write_landmark_rows(writer, name, landmarks)
            io.save_image(draw_overlay(image, landmarks), _overlay_path(output_dir, path))
    finally:
        fh.close()

    rate = (detected / processed) if processed else 0.0
    print("mode: inference")
    print("images processed: %d" % processed)
    print("pose detected:    %d (%.1f%%)" % (detected, 100.0 * rate))
    print("landmarks csv:    %s" % csv_path)
    print("overlays dir:     %s" % os.path.join(output_dir, "overlays"))


def run_validation(image_paths, output_dir, pose, annotations_path):
    """Validation mode: metrics against ground truth."""
    gt = io.load_ground_truth(annotations_path)
    records = []
    missing_gt = 0
    for path in image_paths:
        name = os.path.basename(path)
        if name not in gt:
            missing_gt += 1
            continue
        image = io.load_image(path)
        if image is None:
            print("skip (unreadable): " + path)
            continue
        records.append(evaluate(image, gt[name], pose))

    agg = aggregate(records)
    print("mode: validate")
    print("images with ground truth: %d" % agg["images_total"])
    if missing_gt:
        print("images skipped (no GT):   %d" % missing_gt)
    print("pose detected:            %d (%.1f%%)"
          % (agg["images_detected"], 100.0 * agg["detection_rate"]))
    print("images skipped from PCK (no torso ref): %d" % agg["images_pck_skipped"])
    print("")
    print("%-8s %12s %12s %10s" % ("group", "mean_MPJPE", "mean_PCK@0.2", "n_joints"))
    for g in ("overall", "limb", "face"):
        m = agg["groups"][g]
        mpjpe = "n/a" if m["mpjpe_px"] is None else "%.2f px" % m["mpjpe_px"]
        pck = "n/a" if m["pck"] is None else "%.3f" % m["pck"]
        print("%-8s %12s %12s %10d" % (g, mpjpe, pck, m["n_joints_mpjpe"]))


def main():
    parser = argparse.ArgumentParser(description="neokine pose inference / validation")
    parser.add_argument("--input", required=True, help="folder of RGB images")
    parser.add_argument("--output", default="outputs", help="output folder")
    parser.add_argument("--limit", type=int, default=None, help="process at most N images")
    parser.add_argument("--validate", action="store_true",
                        help="compute PCK/MPJPE against ground truth")
    parser.add_argument("--annotations", default=None,
                        help="COCO keypoint JSON (required with --validate)")
    parser.add_argument("--model-complexity", type=int, default=2, choices=(0, 1, 2))
    args = parser.parse_args()

    if args.validate and not args.annotations:
        parser.error("--validate requires --annotations")

    image_paths = io.list_images(args.input)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        parser.error("no images found in " + args.input)

    pose = create_pose(model_complexity=args.model_complexity)
    try:
        if args.validate:
            run_validation(image_paths, args.output, pose, args.annotations)
        else:
            run_inference(image_paths, args.output, pose)
    finally:
        pose.close()


if __name__ == "__main__":
    main()
