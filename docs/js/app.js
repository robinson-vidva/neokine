// Image + live-webcam + video UI wiring, plus an interactive canvas (zoom/pan,
// hover tooltip, pin, display filters). All state is in memory only; nothing
// is stored, nothing is recorded.

import {
  createLandmarker,
  LANDMARK_NAMES, CONNECTIONS, REGION_COLORS,
  GROUPS, GROUP_OF, LABELABLE, MAJOR, jointColor,
} from "./pose.js";

const el = (id) => document.getElementById(id);
const modelState = el("model-state");
const modelStateLabel = el("model-state-label");
const statusEl = el("status");
const poseCount = el("pose-count");
const variantNote = el("variant-note");
const stage = el("stage");
const canvas = el("output");
const ctx = canvas.getContext("2d");
const video = el("cam");
const vidfile = el("vidfile");
const notice = el("notice");
const placeholderText = el("placeholder-text");
const tooltip = el("tooltip");

// Per-mode variant defaults: full everywhere (deliberate consistency; heavy
// stays reachable in the panel).
const variantByMode = { image: "full", webcam: "full", video: "full" };
const settings = { numPoses: 1, minDetection: 0.5, minPresence: 0.5 };

const VIDEO_CAP = 10.0;
const VIDEO_STEP = 0.1;
const VIDEO_MAX_FRAMES = 100;
// Backstop against a file so large the initial load hangs. Raised from 100MB:
// a <10s 4K clip can exceed 100MB while being perfectly valid, and creating the
// object URL does not decode the whole file (only metadata + sampled seeks), so
// the real protector is the 10s / 100-frame sampling cap, not the byte size.
const VIDEO_MAX_BYTES = 500 * 1024 * 1024;
// Replay frame cache (approach 2): each sampled frame is downscaled to fit this
// longest side and stored as a JPEG blob. Keeps memory to a few MB regardless
// of source resolution, and gives smooth resolution-independent playback.
const REPLAY_MAXDIM = 1000;

let mode = "image";
let landmarker = null;
let builtMode = null;
let builtVariant = null;
let dirty = false;

let lastImage = null;
let stream = null;
let streaming = false;
let rafId = null;

// --- scene + view (interaction) state ---
// scene: the frame currently on the canvas. source is a stable offscreen copy
// (frameCanvas) for interactive modes, or the live <video> for webcam.
let scene = null;
let lastPts0 = null; // canvas-space points of pose 0, for hit-testing
let lastLms0 = null; // raw normalized landmarks of pose 0, for off-frame checks
let view = null;     // { a, e, f } transform: canvas = a*intrinsic + (e,f)
let fitA = 1;        // the "fit to canvas" scale (zoom baseline)
const frameCanvas = document.createElement("canvas");
const fctx = frameCanvas.getContext("2d");

const pins = new Set();
let hoverIndex = -1;
let pointer = null;   // active drag {x,y,e,f}
let panning = false;

// display filter + overlay + interaction state
const groupOn = {};
for (const k of Object.keys(GROUPS)) groupOn[k] = GROUPS[k].on;
const jointOn = {}; // index -> bool (default true when absent)
let labelMode = "hover"; // 'hover' | 'always'
const interactions = { hover: true, visibility: true, pin: true, zoom: true };
let hideOOF = true; // hide landmarks the model places outside the frame (guesses)

function landmarkVisible(i) {
  return groupOn[GROUP_OF[i]] && jointOn[i] !== false;
}
// A landmark whose normalized position falls outside [0,1] is off-frame: the
// model's estimate for a joint the camera cannot see, not an observation.
function offFrame(l) { return l.x < 0 || l.x > 1 || l.y < 0 || l.y > 1; }

function setModelState(state, label) {
  modelState.dataset.state = state;
  modelStateLabel.textContent = label;
}
function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status-item" + (kind ? " " + kind : "");
}
function runningModeFor(m) { return m === "image" ? "IMAGE" : "VIDEO"; }

let lastStreamTs = -1;
function nextStreamTs() {
  const now = Math.round(performance.now());
  lastStreamTs = now > lastStreamTs ? now : lastStreamTs + 1;
  return lastStreamTs;
}

async function ensureLandmarker() {
  const rmode = runningModeFor(mode);
  const variant = variantByMode[mode];
  if (landmarker && builtMode === rmode && builtVariant === variant && !dirty) return true;
  setModelState("loading", "loading model");
  try {
    const next = await createLandmarker({ ...settings, variant }, rmode);
    if (landmarker) landmarker.close();
    landmarker = next;
    builtMode = rmode;
    builtVariant = variant;
    dirty = false;
    setModelState("ready", "ready");
    return true;
  } catch (e) {
    setModelState("error", "error");
    setStatus("Could not load the model. Check your connection and retry. " + e, "error");
    return false;
  }
}

// --- canvas sizing + view transform ---

function sizeCanvas() {
  const r = stage.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width));
  canvas.height = Math.max(1, Math.round(r.height));
}

function fitView() {
  if (!scene) return;
  const M = 16;
  const cw = canvas.width - 2 * M;
  const ch = canvas.height - 2 * M;
  const a = Math.min(cw / scene.iw, ch / scene.ih);
  fitA = a;
  view = { a, e: M + (cw - scene.iw * a) / 2, f: M + (ch - scene.ih * a) / 2 };
}

function ensureFit() { if (!view) fitView(); }

function toCanvasPt(nx, ny) {
  const x = scene.mirror ? 1 - nx : nx;
  return { x: view.e + x * scene.iw * view.a, y: view.f + ny * scene.ih * view.a };
}

// --- compositing ---

function roundRectPath(x, y, w, h, r) {
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); return; }
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

const LFONT = 14, PADX = 7, PADY = 4, GAP = 14;
function rectsOverlap(a, b) {
  return !(a.x + a.w < b.x || b.x + b.w < a.x || a.y + a.h < b.y || b.y + b.h < a.y);
}

function placeLabels(indices, pts, bounds, jointIdx) {
  ctx.font = LFONT + "px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle";
  const placed = [];
  const jointPts = jointIdx.map((i) => pts[i]);
  const minX = Math.max(2, bounds.x + 2), minY = Math.max(2, bounds.y + 2);
  const maxX = Math.min(canvas.width - 2, bounds.x + bounds.w - 2);
  const maxY = Math.min(canvas.height - 2, bounds.y + bounds.h - 2);
  const dirs = [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [1, 1], [0, 1], [-1, 1]];

  for (const i of indices) {
    const p = pts[i];
    if (!p) continue;
    const text = LANDMARK_NAMES[i];
    const w = ctx.measureText(text).width + PADX * 2;
    const h = LFONT + PADY * 2;
    let best = null, leader = false;
    for (let ring = 0; ring < 4 && !best; ring++) {
      const gap = GAP + ring * 18;
      for (const [dx, dy] of dirs) {
        const x = p.x + dx * gap + (dx < 0 ? -w : dx > 0 ? 0 : -w / 2);
        const y = p.y + dy * gap + (dy < 0 ? -h : dy > 0 ? 0 : -h / 2);
        const rect = { x, y, w, h };
        if (rect.x < minX || rect.y < minY || rect.x + w > maxX || rect.y + h > maxY) continue;
        let ok = true;
        for (const r of placed) if (rectsOverlap(rect, r)) { ok = false; break; }
        if (ok) for (const jp of jointPts) {
          if (jp.x > rect.x && jp.x < rect.x + w && jp.y > rect.y && jp.y < rect.y + h && !(jp === p)) { ok = false; break; }
        }
        if (ok) { best = rect; leader = ring > 0 || gap > 20; break; }
      }
    }
    if (!best) {
      // No collision-free candidate around the joint: never drop the label.
      // Stack it into the nearest free vertical slot (on whichever side has
      // room), clamped to the media rect, with a leader line back to the joint.
      const collides = (r) => placed.some((q) => rectsOverlap(r, q));
      const tryColumn = (cx) => {
        for (let off = 0; off <= (maxY - minY); off += h + 3) {
          for (const yy of [p.y - h / 2 + off, p.y - h / 2 - off]) {
            const y = Math.max(minY, Math.min(maxY - h, yy));
            const r = { x: cx, y, w, h };
            if (!collides(r)) return r;
          }
        }
        return null;
      };
      best = tryColumn(Math.min(maxX - w, Math.max(minX, p.x + GAP)))   // to the right
          || tryColumn(Math.max(minX, Math.min(maxX - w, p.x - GAP - w))) // to the left
          || { x: Math.min(maxX - w, Math.max(minX, p.x + GAP)), y: Math.max(minY, Math.min(maxY - h, p.y - h / 2)), w, h };
      leader = true;
    }
    placed.push(best);
    if (leader) {
      const lx = best.x > p.x ? best.x : best.x + best.w;
      ctx.strokeStyle = "rgba(20,23,28,0.55)";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(lx, best.y + best.h / 2); ctx.stroke();
    }
    ctx.fillStyle = "rgba(255,255,255,0.94)";
    ctx.strokeStyle = "rgba(0,0,0,0.20)";
    ctx.lineWidth = 1;
    roundRectPath(best.x, best.y, best.w, best.h, 4);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#14171c";
    ctx.fillText(text, best.x + PADX, best.y + best.h / 2);
  }
}

function composite() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!scene) return;
  ensureFit();
  const { iw, ih, mirror } = scene;
  const { a, e, f } = view;

  if (mirror) {
    ctx.save();
    ctx.translate(e + iw * a, f);
    ctx.scale(-1, 1);
    ctx.drawImage(scene.source, 0, 0, iw * a, ih * a);
    ctx.restore();
  } else {
    ctx.drawImage(scene.source, 0, 0, iw, ih, e, f, iw * a, ih * a);
  }

  const res = scene.result;
  if (!res || !res.landmarks || !res.landmarks.length) { lastPts0 = null; lastLms0 = null; return; }
  lastLms0 = res.landmarks[0];

  // Clip everything below to the media rect: the overlay can never paint onto
  // the stage background, in image, webcam, and video modes alike.
  const media = { x: e, y: f, w: iw * a, h: ih * a };
  ctx.save();
  ctx.beginPath();
  ctx.rect(media.x, media.y, media.w, media.h);
  ctx.clip();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const groups = ["face", "torso", "left", "right"];

  res.landmarks.forEach((lms, poseIdx) => {
    const pts = lms.map((l) => toCanvasPt(l.x, l.y));
    const shown = (i) => landmarkVisible(i) && !(hideOOF && offFrame(lms[i]));

    ctx.strokeStyle = "rgba(15,18,22,0.55)";
    ctx.lineWidth = 6;
    for (const g of groups) for (const [x, y] of CONNECTIONS[g]) {
      if (shown(x) && shown(y)) { ctx.beginPath(); ctx.moveTo(pts[x].x, pts[x].y); ctx.lineTo(pts[y].x, pts[y].y); ctx.stroke(); }
    }
    for (const g of groups) {
      ctx.strokeStyle = REGION_COLORS[g];
      ctx.lineWidth = g === "face" ? 2.5 : 3;
      for (const [x, y] of CONNECTIONS[g]) {
        if (shown(x) && shown(y)) { ctx.beginPath(); ctx.moveTo(pts[x].x, pts[x].y); ctx.lineTo(pts[y].x, pts[y].y); ctx.stroke(); }
      }
    }
    for (let i = 0; i < pts.length; i++) {
      if (!shown(i)) continue;
      const major = MAJOR.has(i);
      ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, major ? 5 : 3, 0, 6.2832);
      ctx.fillStyle = "#fff"; ctx.fill();
      ctx.lineWidth = major ? 2 : 1.2; ctx.strokeStyle = jointColor(i); ctx.stroke();
    }

    if (poseIdx === 0) {
      lastPts0 = pts;
      const highlight = new Set(pins);
      if (hoverIndex >= 0 && shown(hoverIndex)) highlight.add(hoverIndex);
      for (const i of highlight) {
        if (!shown(i)) continue;
        ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 9, 0, 6.2832);
        ctx.lineWidth = 2.5; ctx.strokeStyle = "#d97a00"; ctx.stroke();
      }
      const labels = new Set();
      if (labelMode === "always") for (const i of LABELABLE) if (shown(i)) labels.add(i);
      for (const i of pins) if (shown(i)) labels.add(i);
      if (labels.size) {
        const drawnIdx = [];
        for (let i = 0; i < pts.length; i++) if (shown(i)) drawnIdx.push(i);
        placeLabels([...labels], pts, media, drawnIdx);
      }
    }
  });
  ctx.restore();
}

// --- scene builders ---

function setImageScene(img, result) {
  frameCanvas.width = img.naturalWidth;
  frameCanvas.height = img.naturalHeight;
  fctx.drawImage(img, 0, 0);
  scene = { source: frameCanvas, iw: img.naturalWidth, ih: img.naturalHeight, result, mirror: false, interactive: true };
}

function setVideoFrameScene(result) {
  frameCanvas.width = vidfile.videoWidth;
  frameCanvas.height = vidfile.videoHeight;
  fctx.drawImage(vidfile, 0, 0);
  scene = { source: frameCanvas, iw: vidfile.videoWidth, ih: vidfile.videoHeight, result, mirror: false, interactive: true };
}

// --- interactions ---

function interactive() { return scene && scene.interactive && mode !== "webcam"; }

function eventToCanvas(ev) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (ev.clientX - r.left) * (canvas.width / r.width),
    y: (ev.clientY - r.top) * (canvas.height / r.height),
  };
}

function hitTest(ev) {
  if (!lastPts0 || !lastLms0) return -1;
  const m = eventToCanvas(ev);
  const rad = Math.min(40, Math.max(9, 14 * (view.a / fitA)));
  const mx0 = view.e, my0 = view.f, mx1 = view.e + scene.iw * view.a, my1 = view.f + scene.ih * view.a;
  let best = -1, bd = rad * rad;
  for (let i = 0; i < lastPts0.length; i++) {
    if (!landmarkVisible(i)) continue;
    if (hideOOF && offFrame(lastLms0[i])) continue; // off-frame points are not interactive
    const p = lastPts0[i];
    if (p.x < mx0 || p.x > mx1 || p.y < my0 || p.y > my1) continue; // outside media rect (matches the overlay clip)
    const dx = p.x - m.x, dy = p.y - m.y, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function showTooltip(i, ev) {
  const l = scene.result.landmarks[0][i];
  const ix = Math.round(l.x * scene.iw);
  const iy = Math.round(l.y * scene.ih);
  let html = "<b>" + LANDMARK_NAMES[i] + "</b><br>x " + ix + ", y " + iy + " px";
  if (interactions.visibility) {
    html += "<br>visibility " + l.visibility.toFixed(2) +
            '<br><span class="cav">MediaPipe estimate the point is visible; not accuracy</span>';
  }
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const r = stage.getBoundingClientRect();
  let x = ev.clientX - r.left + 14;
  let y = ev.clientY - r.top + 14;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  if (x + tw > r.width) x = ev.clientX - r.left - tw - 14;
  if (y + th > r.height) y = ev.clientY - r.top - th - 14;
  tooltip.style.left = Math.max(0, x) + "px";
  tooltip.style.top = Math.max(0, y) + "px";
}
function hideTooltip() { tooltip.hidden = true; }

canvas.addEventListener("pointerdown", (ev) => {
  if (!interactive()) return;
  pointer = { x: ev.clientX, y: ev.clientY, e: view.e, f: view.f };
  panning = false;
  canvas.setPointerCapture(ev.pointerId);
});
canvas.addEventListener("pointermove", (ev) => {
  if (!interactive()) return;
  if (pointer && interactions.zoom) {
    const r = canvas.getBoundingClientRect();
    const dx = (ev.clientX - pointer.x) * (canvas.width / r.width);
    const dy = (ev.clientY - pointer.y) * (canvas.height / r.height);
    if (!panning && Math.abs(dx) + Math.abs(dy) > 4) panning = true;
    if (panning) { view.e = pointer.e + dx; view.f = pointer.f + dy; hideTooltip(); composite(); return; }
  }
  if (interactions.hover && !panning) {
    const i = hitTest(ev);
    if (i !== hoverIndex) { hoverIndex = i; composite(); }
    if (i >= 0) showTooltip(i, ev); else hideTooltip();
  }
});
canvas.addEventListener("pointerup", (ev) => {
  if (interactive() && pointer && !panning && interactions.pin) {
    const i = hitTest(ev);
    if (i >= 0) { pins.has(i) ? pins.delete(i) : pins.add(i); composite(); }
  }
  pointer = null; panning = false;
});
canvas.addEventListener("pointerleave", () => {
  if (pointer) return;
  hoverIndex = -1; hideTooltip(); composite();
});
canvas.addEventListener("wheel", (ev) => {
  if (!interactive() || !interactions.zoom) return;
  ev.preventDefault();
  const m = eventToCanvas(ev);
  const z = ev.deltaY < 0 ? 1.1 : 1 / 1.1;
  const na = Math.min(fitA * 12, Math.max(fitA * 0.8, view.a * z));
  const zz = na / view.a;
  view.e = m.x - zz * (m.x - view.e);
  view.f = m.y - zz * (m.y - view.f);
  view.a = na;
  hideTooltip();
  composite();
  updateInteractiveUI();
}, { passive: false });

function updateInteractiveUI() {
  const canInteract = mode !== "webcam";
  el("interactions").classList.toggle("disabled", !canInteract);
  el("webcam-interact-note").hidden = canInteract;
  el("resetview").hidden = !(interactive() && interactions.zoom);
  canvas.style.cursor = interactive() && interactions.zoom ? "grab" : "default";
}

// --- image mode ---

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("could not read image")); };
    img.src = url;
  });
}

let imageRunId = 0;

async function detectImage() {
  if (mode !== "image" || !lastImage) return;
  const myRun = ++imageRunId;
  setModelState("running", "running");
  setStatus("Running pose...");
  if (!(await ensureLandmarker())) return;
  if (myRun !== imageRunId || mode !== "image" || !lastImage) return;
  try {
    const result = landmarker.detect(lastImage);
    setImageScene(lastImage, result);
    composite();
    stage.classList.add("has-image");
    updateInteractiveUI();
    const n = result.landmarks.length;
    poseCount.textContent = "poses: " + n;
    setStatus(n > 0 ? "Pose detected." : "No pose detected in this image.", n > 0 ? "ready" : "error");
  } catch (e) {
    setStatus("Failed to process image. " + e, "error");
  } finally {
    setModelState("ready", "ready");
  }
}

el("image-input").addEventListener("change", async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if (!file) return;
  clearRenderState();
  lastImage = null;
  el("file-name").textContent = file.name;
  try {
    const img = await loadImage(file);
    lastImage = img;
    await detectImage();
  } catch (e) {
    setStatus("Failed to read image. " + e, "error");
  }
});

// --- webcam mode ---

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e) {
    setStatus("Camera unavailable or permission denied. " + e, "error");
    return;
  }
  video.srcObject = stream;
  await video.play();
  if (!(await ensureLandmarker())) { stopCamera(); return; }
  streaming = true;
  el("cam-start").disabled = true;
  el("cam-stop").disabled = false;
  setModelState("running", "running");
  setStatus("Webcam running.", "ready");
  tick();
}

function teardownCamera() {
  streaming = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;
  el("cam-start").disabled = false;
  el("cam-stop").disabled = true;
}

function stopCamera() {
  teardownCamera();
  scene = null; lastPts0 = null;
  clearCanvas();
  stage.classList.remove("has-image");
  poseCount.textContent = "poses: --";
  setModelState("ready", "ready");
  setStatus("Camera stopped.", "ready");
}

async function tick() {
  if (!streaming) return;
  if (dirty || builtVariant !== variantByMode.webcam) {
    if (!(await ensureLandmarker())) { stopCamera(); return; }
    if (!streaming) return;
    setModelState("running", "running");
  }
  if (video.readyState >= 2 && landmarker) {
    try {
      const result = landmarker.detectForVideo(video, nextStreamTs());
      scene = { source: video, iw: video.videoWidth, ih: video.videoHeight, result, mirror: true, interactive: false };
      fitView(); // webcam is always fit (no zoom/pan)
      composite();
      stage.classList.add("has-image");
      poseCount.textContent = "poses: " + result.landmarks.length;
    } catch (e) { /* transient frame errors should not kill the loop */ }
  }
  if (streaming) rafId = requestAnimationFrame(tick);
}

el("cam-start").addEventListener("click", startCamera);
el("cam-stop").addEventListener("click", stopCamera);

// --- video mode (sampled, first 10 seconds) ---

let videoURL = null;
let videoRunId = 0;
let lastVideoFile = null;

// Replay state (approach 2: downscaled JPEG-blob cache of the sampled frames).
let videoFrames = [];   // [{ t, result, blob, w, h }]
let replayIdx = 0;
let replayPlaying = false;
let replayLoop = false;
let replayTimer = null;
let replayBitmap = null; // currently decoded frame (closed when replaced)
let replayToken = 0;     // guards out-of-order async decodes during scrubbing
let replayPlayToken = 0; // ensures at most one play loop exists at a time

function cancelVideo() {
  videoRunId++;
  if (videoURL) { URL.revokeObjectURL(videoURL); videoURL = null; }
  vidfile.removeAttribute("src");
  vidfile.load();
  notice.textContent = "";
}

function waitEvent(target, name) {
  return new Promise((resolve) => {
    const h = () => { target.removeEventListener(name, h); resolve(); };
    target.addEventListener(name, h);
  });
}

function seekTo(t) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; vidfile.removeEventListener("seeked", finish); clearTimeout(to); resolve(); } };
    const to = setTimeout(finish, 2000);
    vidfile.addEventListener("seeked", finish);
    vidfile.currentTime = t;
  });
}

async function processVideoFile(file) {
  cancelVideo();
  clearRenderState();
  const myRun = videoRunId;
  lastVideoFile = file;
  el("video-name").textContent = file.name;

  videoURL = URL.createObjectURL(file);
  vidfile.src = videoURL;
  setStatus("Loading video...");
  if (vidfile.readyState < 2) await waitEvent(vidfile, "loadeddata");
  if (myRun !== videoRunId) return;
  if (!(await ensureLandmarker())) return;
  if (myRun !== videoRunId) return;

  setModelState("running", "running");
  let count = 0;
  let endedEarly = false;
  let t = 0;
  while (t < VIDEO_CAP && count < VIDEO_MAX_FRAMES) {
    if (myRun !== videoRunId) return;
    await seekTo(t);
    if (myRun !== videoRunId) return;
    const actual = vidfile.currentTime;
    if (actual < t - 0.05) { endedEarly = true; break; }
    let frameResult = null;
    try {
      frameResult = landmarker.detectForVideo(vidfile, nextStreamTs());
      setVideoFrameScene(frameResult);
      composite();
      stage.classList.add("has-image");
      updateInteractiveUI();
      poseCount.textContent = "poses: " + frameResult.landmarks.length;
    } catch (e) { /* skip a bad frame */ }
    if (frameResult) {
      const cap = await frameToBlob(vidfile, vidfile.videoWidth, vidfile.videoHeight);
      videoFrames.push({ t: actual, result: frameResult, blob: cap.blob, w: cap.w, h: cap.h });
    }
    count++;
    const pct = Math.min(100, Math.round((t / VIDEO_CAP) * 100));
    setStatus("Processing video... " + actual.toFixed(1) + "s (" + pct + "%)");
    await new Promise((r) => setTimeout(r, 0));
    t = Math.round((t + VIDEO_STEP) * 1000) / 1000;
  }
  if (myRun !== videoRunId) return;

  let longer = false;
  if (!endedEarly) {
    if (isFinite(vidfile.duration) && vidfile.duration > VIDEO_CAP + 0.05) longer = true;
    else {
      await seekTo(VIDEO_CAP + 0.3);
      if (myRun !== videoRunId) return;
      if (vidfile.currentTime > VIDEO_CAP + 0.05) longer = true;
    }
  }
  notice.textContent = longer ? "Showing first 10 seconds" : "";

  // Frames are cached as blobs; the video element/URL is no longer needed for
  // replay (approach 2). Free it now; resetWorkspace revokes again as a no-op.
  if (videoURL) { URL.revokeObjectURL(videoURL); videoURL = null; }
  vidfile.removeAttribute("src");
  vidfile.load();
  if (videoFrames.length) { replayIdx = videoFrames.length - 1; await showReplayFrame(replayIdx); }
  updateReplayUI();

  setModelState("ready", "ready");
  setStatus("Processed " + (longer ? "first 10 seconds" : "video") + " (" + count + " frames sampled).", "ready");
}

// --- replay (sampled playback of the cached frames) ---

function frameToBlob(src, iw, ih) {
  const s = Math.min(1, REPLAY_MAXDIM / Math.max(iw, ih));
  const w = Math.max(1, Math.round(iw * s)), h = Math.max(1, Math.round(ih * s));
  const oc = document.createElement("canvas");
  oc.width = w; oc.height = h;
  oc.getContext("2d").drawImage(src, 0, 0, w, h);
  return new Promise((res) => oc.toBlob((b) => res({ blob: b, w, h }), "image/jpeg", 0.82));
}

async function showReplayFrame(k) {
  if (!videoFrames.length) return;
  k = Math.max(0, Math.min(videoFrames.length - 1, k));
  replayIdx = k;
  const fr = videoFrames[k];
  const my = ++replayToken;
  const bmp = await createImageBitmap(fr.blob);
  if (my !== replayToken) { bmp.close(); return; } // a newer frame request superseded this one
  if (replayBitmap) replayBitmap.close();
  replayBitmap = bmp;
  scene = { source: bmp, iw: fr.w, ih: fr.h, result: fr.result, mirror: false, interactive: true };
  composite(); // preserves the current zoom/pan (view) across frames
  stage.classList.add("has-image");
  poseCount.textContent = "poses: " + fr.result.landmarks.length;
  updateReplayUI();
}

function pauseReplay() {
  replayPlaying = false;
  replayPlayToken++; // invalidate any in-flight play loop
  if (replayTimer) { clearTimeout(replayTimer); replayTimer = null; }
  updateReplayUI();
}

function playReplay() {
  if (!videoFrames.length) return;
  const myPlay = ++replayPlayToken; // a fresh loop; supersedes any prior one
  if (replayIdx >= videoFrames.length - 1) replayIdx = -1; // restart from the beginning
  replayPlaying = true;
  updateReplayUI();
  const step = async () => {
    if (!replayPlaying || myPlay !== replayPlayToken) return;
    let next = replayIdx + 1;
    if (next >= videoFrames.length) {
      if (replayLoop) next = 0;
      else { replayPlaying = false; updateReplayUI(); return; }
    }
    await showReplayFrame(next);
    // Re-check the token after the async decode: if a pause/play happened in
    // that window, this stale loop must not reschedule.
    if (replayPlaying && myPlay === replayPlayToken) replayTimer = setTimeout(step, 100); // ~10 fps
  };
  replayTimer = setTimeout(step, 100);
}

function updateReplayUI() {
  const M = videoFrames.length;
  el("replay").hidden = M === 0;
  el("rp-play").textContent = replayPlaying ? "Pause" : "Play";
  const sc = el("rp-scrub");
  sc.max = String(Math.max(0, M - 1));
  sc.value = String(replayIdx);
  el("rp-loop").checked = replayLoop;
  const t = M ? videoFrames[replayIdx].t : 0;
  el("rp-counter").textContent = M ? "frame " + (replayIdx + 1) + " / " + M + "  @ " + t.toFixed(2) + "s" : "frame 0 / 0";
}

function stopReplay() {
  pauseReplay();
  replayToken++;
  if (replayBitmap) { replayBitmap.close(); replayBitmap = null; }
  videoFrames = [];
  replayIdx = 0;
  el("replay").hidden = true;
  el("rp-scrub").value = "0"; el("rp-scrub").max = "0";
  el("rp-counter").textContent = "frame 0 / 0";
  el("rp-play").textContent = "Play";
}

el("rp-play").addEventListener("click", () => { replayPlaying ? pauseReplay() : playReplay(); });
el("rp-back").addEventListener("click", () => { pauseReplay(); showReplayFrame(replayIdx - 1); });
el("rp-fwd").addEventListener("click", () => { pauseReplay(); showReplayFrame(replayIdx + 1); });
el("rp-loop").addEventListener("change", (e) => { replayLoop = e.target.checked; });
el("rp-scrub").addEventListener("input", (e) => { const k = Number(e.target.value); pauseReplay(); showReplayFrame(k); });

el("video-input").addEventListener("change", (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if (!file) return;
  if (file.size > VIDEO_MAX_BYTES) {
    cancelVideo();
    clearRenderState();
    el("video-name").textContent = file.name;
    setStatus("File too large (" + (file.size / 1048576).toFixed(0) + " MB). Limit is 100 MB.", "error");
    return;
  }
  processVideoFile(file);
});

// --- teardown / reset ---

function clearCanvas() { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height); }

// Clear the rendered output + interaction state (canvas, scene, view, pins,
// hover). Used when loading a new file within a mode.
function clearRenderState() {
  imageRunId++;
  stopReplay();
  scene = null;
  lastPts0 = null;
  lastLms0 = null;
  view = null;
  pins.clear();
  hoverIndex = -1;
  panning = false; pointer = null;
  hideTooltip();
  clearCanvas();
  stage.classList.remove("has-image");
  poseCount.textContent = "poses: --";
  updateInteractiveUI();
}

// The one routine that returns the whole workspace to a clean slate.
function resetWorkspace() {
  clearTimeout(rerunTimer);
  teardownCamera();
  cancelVideo();
  clearRenderState();
  lastImage = null;
  lastVideoFile = null;
  el("image-input").value = "";
  el("video-input").value = "";
  el("file-name").textContent = "no file";
  el("video-name").textContent = "no file";
  notice.textContent = "";
}

// --- mode switch ---

function updateVariantUI() {
  const v = variantByMode[mode];
  for (const b of el("variant").querySelectorAll("button")) {
    b.setAttribute("aria-checked", String(b.dataset.variant === v));
  }
  variantNote.hidden = !((mode === "webcam" || mode === "video") && v === "heavy");
}

const PLACEHOLDER = {
  image: "Choose an image to run pose",
  webcam: "Start camera to run pose",
  video: "Upload a video to process",
};
const MODE_STATUS = {
  image: "Image mode. Choose an image.",
  webcam: "Webcam mode. Start the camera.",
  video: "Video mode. Upload a video (first 10 seconds are processed).",
};

function switchMode(next) {
  if (next === mode) return;
  resetWorkspace();
  mode = next;
  document.body.dataset.mode = mode;
  for (const b of document.querySelectorAll(".mode-btn[data-mode]")) b.classList.toggle("is-active", b.dataset.mode === mode);
  for (const p of document.querySelectorAll(".mode-panel")) p.hidden = p.dataset.for !== mode;
  placeholderText.textContent = PLACEHOLDER[mode];
  updateVariantUI();
  updateInteractiveUI();
  setModelState("ready", "ready");
  setStatus(MODE_STATUS[mode], "ready");
}

for (const b of document.querySelectorAll(".mode-btn[data-mode]")) {
  b.addEventListener("click", () => { if (!b.disabled) switchMode(b.dataset.mode); });
}

// --- model + settings controls ---

el("variant").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-variant]");
  if (!btn || btn.dataset.variant === variantByMode[mode]) return;
  variantByMode[mode] = btn.dataset.variant;
  updateVariantUI();
  dirty = true;
  if (mode === "image") { const ok = await ensureLandmarker(); if (ok && lastImage) detectImage(); }
  else if (mode === "video" && lastVideoFile) processVideoFile(lastVideoFile);
});

let rerunTimer = null;
function scheduleRerun() {
  clearTimeout(rerunTimer);
  rerunTimer = setTimeout(() => {
    if (mode === "image" && lastImage) detectImage();
    else if (mode === "video" && lastVideoFile) processVideoFile(lastVideoFile);
  }, 250);
}

el("numposes").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-step]");
  if (!btn) return;
  settings.numPoses = Math.max(1, Math.min(4, settings.numPoses + Number(btn.dataset.step)));
  el("numposes-val").textContent = String(settings.numPoses);
  dirty = true;
  scheduleRerun();
});

function bindSlider(id, valId, key) {
  const slider = el(id), out = el(valId);
  slider.addEventListener("input", () => {
    settings[key] = Number(slider.value);
    out.textContent = settings[key].toFixed(2);
    dirty = true;
    scheduleRerun();
  });
}
bindSlider("det", "det-val", "minDetection");
bindSlider("pres", "pres-val", "minPresence");

// label mode
for (const r of document.querySelectorAll('input[name="labelmode"]')) {
  r.addEventListener("change", () => { if (r.checked) { labelMode = r.value; composite(); } });
}

// interaction toggles
function bindInteraction(id, key) {
  el(id).addEventListener("change", (ev) => {
    interactions[key] = ev.target.checked;
    if (!interactions.hover) hideTooltip();
    updateInteractiveUI();
    composite();
  });
}
bindInteraction("int-hover", "hover");
bindInteraction("int-vis", "visibility");
bindInteraction("int-pin", "pin");
bindInteraction("int-zoom", "zoom");

el("hide-oof").addEventListener("change", (ev) => {
  hideOOF = ev.target.checked;
  if (hoverIndex >= 0 && lastLms0 && hideOOF && offFrame(lastLms0[hoverIndex])) { hoverIndex = -1; hideTooltip(); }
  composite();
});

el("resetview").addEventListener("click", () => { if (scene) { fitView(); composite(); updateInteractiveUI(); } });

// display-filter checkboxes (generated) + master toggle with indeterminate state
const filterCtl = { groups: {}, mains: {} };

// A group is 'on' (all its points shown), 'off' (group hidden), or 'partial'
// (group on but some main sub-joints unticked -> indeterminate).
function groupState(key) {
  if (!groupOn[key]) return "off";
  const mains = GROUPS[key].mains;
  if (!mains.length) return "on";
  const onCount = mains.filter((i) => jointOn[i] !== false).length;
  return onCount === mains.length ? "on" : "partial";
}
function masterState() {
  const states = Object.keys(GROUPS).map(groupState);
  if (states.every((s) => s === "on")) return "on";
  if (states.every((s) => s === "off")) return "off";
  return "partial";
}
function syncFilterUI() {
  for (const key of Object.keys(GROUPS)) {
    const st = groupState(key);
    const cb = filterCtl.groups[key];
    if (cb) { cb.checked = st === "on"; cb.indeterminate = st === "partial"; }
    for (const idx of GROUPS[key].mains) {
      const mc = filterCtl.mains[idx];
      if (mc) mc.checked = jointOn[idx] !== false;
    }
  }
  const ms = masterState();
  const master = el("filter-all");
  master.checked = ms === "on";
  master.indeterminate = ms === "partial";
}
function showAllPoints() {
  for (const k of Object.keys(GROUPS)) groupOn[k] = true;
  for (const k of Object.keys(jointOn)) delete jointOn[k];
  syncFilterUI(); composite();
}
function hideAllPoints() {
  for (const k of Object.keys(GROUPS)) groupOn[k] = false;
  syncFilterUI(); composite();
}

function buildFilters() {
  const host = el("filters");
  host.innerHTML = "";
  filterCtl.groups = {}; filterCtl.mains = {};
  for (const key of Object.keys(GROUPS)) {
    const g = GROUPS[key];
    const row = document.createElement("div");
    row.className = "filter-group";
    const gl = document.createElement("label");
    gl.className = "check";
    const gc = document.createElement("input");
    gc.type = "checkbox";
    gc.addEventListener("change", () => {
      groupOn[key] = gc.checked;
      if (gc.checked) for (const i of g.mains) delete jointOn[i]; // checking a group shows all its points
      syncFilterUI(); composite();
    });
    filterCtl.groups[key] = gc;
    gl.appendChild(gc);
    const gs = document.createElement("span"); gs.textContent = g.label; gl.appendChild(gs);
    row.appendChild(gl);
    if (g.mains.length) {
      const sub = document.createElement("div");
      sub.className = "filter-mains";
      for (const idx of g.mains) {
        const ml = document.createElement("label");
        ml.className = "check small";
        const mc = document.createElement("input");
        mc.type = "checkbox";
        mc.addEventListener("change", () => { jointOn[idx] = mc.checked; syncFilterUI(); composite(); });
        filterCtl.mains[idx] = mc;
        ml.appendChild(mc);
        const msp = document.createElement("span"); msp.textContent = LANDMARK_NAMES[idx]; ml.appendChild(msp);
        sub.appendChild(ml);
      }
      row.appendChild(sub);
    }
    host.appendChild(row);
  }
  syncFilterUI();
}

// Master toggle: checked -> show all; unchecked -> hide all. Clicking while
// indeterminate sets checked=true (browser default) -> show all.
el("filter-all").addEventListener("change", (ev) => {
  if (ev.target.checked) showAllPoints(); else hideAllPoints();
});

// Update / Reset
function applyNow() {
  dirty = true;
  if (mode === "image" && lastImage) detectImage();
  else if (mode === "video" && lastVideoFile) processVideoFile(lastVideoFile);
}
function resetSettings() {
  variantByMode.image = "full"; variantByMode.webcam = "full"; variantByMode.video = "full";
  settings.numPoses = 1; settings.minDetection = 0.5; settings.minPresence = 0.5;
  labelMode = "hover";
  for (const k of Object.keys(GROUPS)) groupOn[k] = GROUPS[k].on;
  for (const k of Object.keys(jointOn)) delete jointOn[k];
  interactions.hover = interactions.visibility = interactions.pin = interactions.zoom = true;
  hideOOF = true;
  pins.clear(); hoverIndex = -1;

  el("numposes-val").textContent = "1";
  el("det").value = "0.5"; el("det-val").textContent = "0.50";
  el("pres").value = "0.5"; el("pres-val").textContent = "0.50";
  document.querySelector('input[name="labelmode"][value="hover"]').checked = true;
  el("int-hover").checked = el("int-vis").checked = el("int-pin").checked = el("int-zoom").checked = true;
  el("hide-oof").checked = true;
  buildFilters();
  updateVariantUI();
  updateInteractiveUI();

  dirty = true;
  if (mode === "image" && lastImage) detectImage();
  else if (mode === "video" && lastVideoFile) processVideoFile(lastVideoFile);
  else composite();
}
el("apply-btn").addEventListener("click", applyNow);
el("reset-btn").addEventListener("click", resetSettings);

window.addEventListener("pagehide", () => { teardownCamera(); cancelVideo(); });

const ro = new ResizeObserver(() => { sizeCanvas(); if (scene) { fitView(); composite(); } });
ro.observe(stage);

// initial load
sizeCanvas();
buildFilters();
updateVariantUI();
updateInteractiveUI();
setStatus("Loading pose model...");
ensureLandmarker().then((ok) => { if (ok) setStatus("Pose model ready. Choose an image.", "ready"); });
