// MediaPipe Tasks Vision PoseLandmarker (body-only pose; NOT Holistic).
// WASM fileset and the .task model load from Google's CDN in the user's
// browser at runtime. No storage, no backend. This module holds the model
// factory plus the static skeleton/label constants; all drawing and
// interaction live in app.js.

// The MediaPipe library is imported lazily (dynamic import) rather than as a
// top-level static import. A static import that stalls or errors (as it can in
// Safari) would prevent this whole module - and therefore the entire app - from
// ever running, leaving a silent "loading" state. Lazy + timed loading keeps
// the UI alive and surfaces any failure as a visible error.
const TASKS_VISION_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

let mpLib = null;
async function loadMP(onStage) {
  if (!mpLib) {
    if (onStage) onStage("loading MediaPipe library");
    mpLib = await withTimeout(import(TASKS_VISION_URL), 20000, "MediaPipe library");
  }
  return mpLib;
}

export const MODEL_URLS = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
  heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
};

export const LANDMARK_NAMES = [
  "nose", "left eye (inner)", "left eye", "left eye (outer)",
  "right eye (inner)", "right eye", "right eye (outer)", "left ear",
  "right ear", "mouth (left)", "mouth (right)", "left shoulder",
  "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
  "left pinky", "right pinky", "left index", "right index", "left thumb",
  "right thumb", "left hip", "right hip", "left knee", "right knee",
  "left ankle", "right ankle", "left heel", "right heel",
  "left foot index", "right foot index",
];

// Connections grouped by region so each is colored distinctly. A connection is
// drawn only when both endpoints are currently visible (display filter).
export const CONNECTIONS = {
  face: [[0, 2], [2, 7], [0, 5], [5, 8], [9, 10]],
  torso: [[11, 12], [11, 23], [12, 24], [23, 24]],
  left: [[11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
         [23, 25], [25, 27], [27, 29], [29, 31], [27, 31]],
  right: [[12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
          [24, 26], [26, 28], [28, 30], [30, 32], [28, 32]],
};

export const REGION_COLORS = {
  left: "#12b886",   // green
  right: "#f59f00",  // amber
  torso: "#4c6ef5",  // indigo
  face: "#adb5bd",   // gray
};

// Display-filter groups. `members` = every landmark in the group; `mains` =
// the joints that get an individual on/off toggle and a text label. Face is
// off by default (its connections read as a distracting mask, not body pose).
export const GROUPS = {
  face:  { label: "Face",        members: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mains: [0, 2, 5, 7, 8], on: false },
  torso: { label: "Torso",       members: [11, 12, 23, 24],                   mains: [11, 12, 23, 24], on: true },
  left:  { label: "Left limbs",  members: [13, 15, 25, 27],                   mains: [13, 15, 25, 27], on: true },
  right: { label: "Right limbs", members: [14, 16, 26, 28],                   mains: [14, 16, 26, 28], on: true },
  hands: { label: "Hands",       members: [17, 18, 19, 20, 21, 22],           mains: [], on: true },
  feet:  { label: "Feet",        members: [29, 30, 31, 32],                   mains: [], on: true },
};

// index -> group key
export const GROUP_OF = (() => {
  const arr = new Array(33);
  for (const key of Object.keys(GROUPS)) for (const i of GROUPS[key].members) arr[i] = key;
  return arr;
})();

// Joints eligible for a text label (the main joints across all groups).
export const LABELABLE = [
  ...GROUPS.face.mains, ...GROUPS.torso.mains, ...GROUPS.left.mains, ...GROUPS.right.mains,
];

// Joints drawn larger.
export const MAJOR = new Set([0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]);

export function jointColor(i) {
  if (i <= 10) return REGION_COLORS.face;
  return i % 2 === 1 ? REGION_COLORS.left : REGION_COLORS.right;
}

let vision = null;
async function getVision(onStage) {
  const { FilesetResolver } = await loadMP(onStage);
  if (!vision) {
    if (onStage) onStage("loading vision runtime (WASM)");
    // ~9 MB WASM from the CDN; time out rather than hang forever if it stalls.
    vision = await withTimeout(FilesetResolver.forVisionTasks(WASM_BASE), 30000, "WASM runtime");
  }
  return vision;
}

// onStage(message) is optional; it surfaces which step is in progress so a
// stall is visible on screen instead of a generic "loading model".
export async function createLandmarker(settings, runningMode, onStage) {
  const { PoseLandmarker } = await loadMP(onStage);
  const v = await getVision(onStage);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URLS[settings.variant], delegate },
    runningMode,
    numPoses: settings.numPoses,
    minPoseDetectionConfidence: settings.minDetection,
    minPosePresenceConfidence: settings.minPresence,
    minTrackingConfidence: 0.5,
  });
  // Try the GPU delegate first, but cap it: on some browsers (notably Safari)
  // GPU init can hang forever instead of rejecting, which would leave the app
  // stuck on "loading model". If it does not resolve in time, or it rejects,
  // fall back to the CPU delegate (slower but reliable everywhere).
  try {
    if (onStage) onStage("loading " + settings.variant + " model (GPU)");
    return await withTimeout(PoseLandmarker.createFromOptions(v, opts("GPU")), 6000, "GPU model");
  } catch (e) {
    if (onStage) onStage("GPU unavailable; loading " + settings.variant + " model (CPU)");
    return await withTimeout(PoseLandmarker.createFromOptions(v, opts("CPU")), 30000, "CPU model");
  }
}

function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error((label || "init") + " timed out after " + (ms / 1000) + "s")), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
