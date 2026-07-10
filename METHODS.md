# Methods & limitations

_Last updated: 2026-07-10_

This document describes exactly what neokine computes and how, so results can be
interpreted and reproduced. **neokine is a visualization demo, not a validated
measurement instrument** (see [Terms](TERMS.md)).

## Pose estimation

- **Model:** Google **MediaPipe Tasks Vision — Pose Landmarker**, the 33-point
  full-body model (BlazePose GHUM topology). Runs **client-side** in the browser
  via a WebAssembly runtime; no server-side inference.
- **Library:** `@mediapipe/tasks-vision@0.10.14` (loaded from a CDN).
- **Model variants:** `lite` / `full` / `heavy` (float16, model version `1`,
  from Google's public model repository). "full" is the default;
  larger variants are more accurate but slower.
- **Confidence controls:** the "Detection threshold" and "Landmark confidence"
  sliders map to MediaPipe's `minPoseDetectionConfidence` and
  `minPosePresenceConfidence`. They change what is shown, not the underlying
  accuracy.
- **Scope:** metrics use the first detected person (`landmarks[0]`). Additional
  people, if any, are drawn but not analyzed.

## Coordinate system

All positions are in **normalized image units**: `x` and `y` in `[0, 1]`, i.e.
fractions of the frame width and height. There is **no camera calibration,
depth, or physical scale**, so values cannot be converted to centimetres,
degrees of true joint rotation, or metres/second.

## Kinematics (video mode)

For a joint tracked across the sampled frames with time `t` and normalized
position `(x, y)`:

- **Displacement (path length):** the sum of straight-line distances between
  consecutive frames — `Σ hypot(xᵢ − xᵢ₋₁, yᵢ − yᵢ₋₁)`, in normalized units.
- **Mean velocity:** path length ÷ elapsed time (`t_last − t_first`), in
  normalized units per second.
- **Left–right asymmetry index:** `(L − R) / (L + R)`, where `L` and `R` are the
  total path lengths of the chosen left/right pair. Range `−1 … 1`; `0` is
  symmetric, positive means the left side moved more.
- **Joint angles:** the interior angle at a joint from three landmarks
  (e.g. shoulder–elbow–wrist), computed in the **2D image plane**. `0°` = fully
  folded, `180°` = straight. The reported **range** is the min–max over the clip.
- **Optional smoothing:** a 3-frame moving average applied to `x`/`y` before the
  velocity calculation (off by default; it changes the velocity estimate).

**Sampling:** in video mode only the first *N* seconds (duration cap) are
processed, sampled at a chosen frame rate (default ~10 fps). Frames are sampled,
not continuous, and are held in memory and discarded — the source video is never
stored.

## Validation

The displayed numbers were checked two independent ways:

1. **Known-truth unit tests** of the math functions (joint angle, path length,
   mean velocity, asymmetry) against hand-computed values on synthetic geometry.
2. **Self-consistency:** after processing a clip, every displayed value was
   independently recomputed from the exported per-frame landmark CSV and matched
   the on-screen figures (and the CSV's own angle columns) to rounding.

This validates that **the math and the pipeline are correct** — that neokine
faithfully reports the kinematics of the landmarks it was given. It does **not**
validate the pose model's accuracy at locating those landmarks.

## Limitations & sources of error

- **2D only.** Motion toward/away from the camera and out-of-plane rotation are
  under- or mis-measured; joint angles are projected, not anatomical.
- **No scale.** Results are relative; comparisons *within one clip* are more
  meaningful than absolute numbers, and two clips are only comparable if framing
  and camera distance match.
- **Camera motion** is indistinguishable from subject motion.
- **Sampling** can alias fast movement (motion between sampled frames is missed).
- **Distal joints** (wrists, ankles, feet) carry the largest positional error,
  so their velocity is the least reliable.
- **Model error propagates.** If the model mis-locates a joint, the kinematics
  faithfully reflect that error.
- **Lighting, occlusion, unusual poses, and infants** (the model is trained
  predominantly on adults) all reduce landmark accuracy.

## Reproducibility

Given the same input and settings, results are deterministic. The library and
model versions are pinned above. The full source is open under the
[Apache License 2.0](LICENSE); see the repository to inspect or run it locally.

## References

- Bazarevsky et al., *BlazePose: On-device Real-time Body Pose Tracking* (2020),
  arXiv:2006.10204.
- MediaPipe Pose Landmarker documentation and model card, Google
  (ai.google.dev/edge/mediapipe).
