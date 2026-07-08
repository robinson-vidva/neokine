# FiDIP deferred run recipe

Setup-and-run recipe for running FiDIP (Ostadabbas infant-specialized pose) as
a challenger to the MediaPipe baseline. Deferred from the Apple Silicon dev
machine (no CUDA, weights would not download, disk 95% full) to a Windows/CUDA
machine. Nothing here needs to be re-derived; follow top to bottom.

Repo: https://github.com/ostadabbas/Infant-Pose-Estimation
Paper: Huang, Fu, Liu, Ostadabbas, "Invariant Representation Learning for
Infant Pose Estimation with Small Data", IEEE FG 2021. arXiv:2010.06100

## 1. Environment FiDIP needs

- OS: Ubuntu 18.04 (repo author config). Windows + CUDA works if the CUDA
  toolchain and a compatible PyTorch build are installed; WSL2 Ubuntu is the
  lowest-friction option on a Windows box.
- Python: 3.12
- CUDA: 12.1 (env is pinned to cuda-cudart 12.1.105, libcublas 12.1, etc.)
- GPU: NVIDIA required. Author tested on TITAN Xp. HRNet-W48 at 384x288.
- Env creation: `conda env create -f fidip_env.yml` then `conda activate fidip`.
- Native build step: `cd ${POSE_ROOT}/lib && make` (builds the repo C
  extensions; DarkPose/HRNet lineage from deep-high-resolution-net).

fidip_env.yml is a fully-pinned linux-64 + CUDA 12.1 conda lock. Channels:
`pytorch`, `nvidia`, `defaults`. Key pinned deps observed in Stage A:

    cuda-cudart=12.1.105        cuda-runtime=12.1.0     cuda-nvrtc=12.1.105
    cuda-libraries=12.1.0       libcublas=12.1.0.26     libcufft=11.0.2.4
    libcusolver=11.4.4.55       libcusparse=12.0.2.55   libcurand=10.3.6.39
    libnpp=12.0.2.50            libnvjpeg=12.1.1.14     libnvjitlink=12.1.105
    ffmpeg=4.3                  libprotobuf=3.20.3      libtiff=4.5.1
    ld_impl_linux-64=2.38       libgcc-ng=11.2.0        libstdcxx-ng=11.2.0

These are Linux/NVIDIA-only builds (no osx-arm64 equivalents) - which is why
the env cannot be instantiated on the Mac. Reuse the yml as-authored on the
CUDA box; do not hand-edit versions. If conda solve fails, fetch a fresh
`fidip_env.yml` from the repo root rather than pinning by hand.

HRNet/DarkPose specifics: top-down model, HRNet-W48 backbone, 384x288 input,
DarkPose decoding. Model config lives under the repo `experiments/` /
`lib/config` (COCO w48_384x288). This is NOT an mmpose install; it is the
deep-high-resolution-net code style bundled in the repo (its own `lib`).

## 2. Weights

All weights are on Google Drive and did NOT download from the headless Mac
(direct file link redirects to a Google sign-in page, returns HTML not the
.pth). On the CUDA machine, download via a browser (authenticated) or gdown.

Files needed (place under `${POSE_ROOT}/models/`):

| file | backbone | source (Drive folder / link) | notes |
|---|---|---|---|
| hrnet_fidip.pth | HRNet-W48 | Drive folder "FiDIP_models" | primary FiDIP model to evaluate |
| mobile_fidip.pth | MobileNetV2 | Drive folder "FiDIP_models" | optional lighter variant |
| pose_hrnet_w48_384x288.pth (COCO DarkPose) | HRNet-W48 | Drive folder "TGA_models" | DarkPose base / imagenet-COCO pretrain |
| new FiDIP (expanded SyRIP) | HRNet-W48 | https://drive.google.com/file/d/1UHQC63mSEL4Vcuenqg9R3OaNP0Wv_6gd/view | trained on expanded SyRIP |

- File sizes were not published; expect HRNet-W48 checkpoints ~250 MB each.
  Verify a few hundred MB downloaded, not a small HTML error page.
- Expanded SyRIP dataset (extra 400 real train images) is PASSWORD-GATED:
  "contact us for the password" (xhuang@ece.neu.edu / ostadabbas@ece.neu.edu).
  Not required to evaluate on validate500/validate100 - only if retraining or
  using the expanded set.
- The gdown fallback for the direct file: `pip install gdown` then
  `gdown 1UHQC63mSEL4Vcuenqg9R3OaNP0Wv_6gd`. Folder links may need
  `gdown --folder <url>`; watch for quota/confirm-token failures.

## 3. FiDIP input and output format, mapping to GT

- Input: top-down. One person crop per image, resized to 384x288, normalized
  (ImageNet mean/std). A person bbox is required; for a clean baseline
  comparison feed the SAME GT bbox used elsewhere (validate annotations carry
  `bbox` = [x, y, w, h], absolute pixels). Decode heatmaps with DarkPose to
  get keypoint pixel coords, then map back to original-image coordinates.
- Output: 17 keypoints in standard COCO-17 order:
      0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
      5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
      9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
      13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
- Mapping to our GT: IDENTITY. FiDIP output order == SyRIP GT COCO-17 order.
  No remap needed (unlike MediaPipe-33, which required a 17-of-33 selection).
  Just decode heatmap coords to original pixels and compare index-for-index.

## 4. Validation protocol to reuse (must match the MediaPipe run exactly)

- Sets: SyRIP validate_infant images with annotations
  `annotations/validate500/...json` (500 real) and
  `annotations/validate100/...json` (100 hard-pose subset).
  Already extracted locally at `dataset/SyRIP_unzipped/SyRIP/`.
- Score only GT joints with visibility == 2. Ignore visibility 0 and 1.
- Metric 1 - MPJPE: mean Euclidean pixel error over visibility==2 joints,
  in ORIGINAL-image coordinates (map any crop-space prediction back first).
- Metric 2 - PCK@0.2, torso-normalized:
    torso length = mean of left and right shoulder-hip distances, using only
    sides where BOTH shoulder and hip are visibility == 2; if one side is
    unusable use the other; if neither is usable, SKIP that image from PCK and
    report the skip count. Threshold = 0.2 * torso. A joint is correct if its
    pixel error <= threshold.
- Breakouts to report (same as MediaPipe run):
    limb (shoulders, elbows, wrists, hips, knees, ankles) vs
    face (nose, eyes, ears). Limb is the headline.
    Per-joint-group: shoulders, elbows, wrists, hips, knees, ankles.
- Coverage to report: detection rate, detection-failure rate, PCK-skips.
- The metric functions in this repo's `core/metrics.py`
  (torso_length, per_joint_errors, aggregate; PCK_ALPHA=0.2, VIS_LABELED=2)
  are model-agnostic and can be reused directly: build a (17,2) predicted
  pixel array in COCO order and feed the same evaluate/aggregate path.

## 5. MediaPipe baseline to beat (reference inline)

Config: mp.solutions.pose (legacy), static_image_mode=True,
model_complexity=2, confidence thresholds at library defaults 0.5/0.5.

| set | limb PCK@0.2 | limb MPJPE (px) | detection rate |
|---|---|---|---|
| validate500 | 0.742 | 36.39 | 94.6% |
| validate100 | 0.749 | 39.74 | 98.0% |

Durable weak spot across both sets: HIPS, PCK ~0.58 (validate500 hips PCK
0.582 / MPJPE 42.29; lower limbs generally weakest, ankles highest error).

KEY QUESTION the FiDIP run must answer: does an infant-specialized model fix
the hips / lower-limb gap? Report FiDIP limb PCK/MPJPE and per-joint-group
side by side with the table above, and state plainly the hips PCK delta and
any cost in detection rate or inference speed.

## Notes / gotchas carried from Stage A

- Weights are the blocker, not the code: confirm each .pth is a real
  multi-hundred-MB file before running.
- Feed GT bbox for the apples-to-apples comparison; if using a person
  detector instead, label results as detector-fed (not GT-bbox) since that
  changes coverage.
- Keep FiDIP fully isolated (its own conda env); it must not touch this repo's
  mediapipe `.venv` or `core/`.
