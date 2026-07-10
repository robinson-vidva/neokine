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

// Sampling knobs (user-selectable). Duration cap in seconds and sampling rate
// in fps; total frames hard-capped regardless of selection.
let videoDurationSec = 10;   // 10 / 15 / 30
let videoFps = 10;           // 10 / 15 / 30
const VIDEO_MAX_FRAMES = 900;      // hard cap (900 seeks ~ 20-60s)
const VIDEO_HEAVY_FRAMES = 300;    // warn above this
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
let viewFitDims = null; // dims the current view was fit for; refit when they change
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
    const next = await createLandmarker({ ...settings, variant }, rmode, (msg) => setStatus(msg + "..."));
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

// Main-canvas size in CSS pixels. ALL app coordinates (view, toCanvasPt, hit
// testing, labels, clip) live in this CSS-px space; the backing store below is
// dpr-scaled only so the overlay is crisp on retina.
let cssW = 0, cssH = 0;
// Clamp so a dpr-3 phone does not allocate an enormous backing store.
function dprValue() { return Math.min(3, Math.max(1, window.devicePixelRatio || 1)); }

function sizeCanvas() {
  const r = stage.getBoundingClientRect();
  cssW = Math.max(1, Math.round(r.width));
  cssH = Math.max(1, Math.round(r.height));
  const dpr = dprValue();
  // Backing store in DEVICE px; CSS display size stays the stage size via the
  // `.stage canvas { width:100%; height:100% }` rule (we never set style.width).
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));
}

function fitView() {
  if (!scene) return;
  const M = 16;
  const cw = cssW - 2 * M;   // CSS px (fit is computed in CSS-px app space)
  const ch = cssH - 2 * M;
  const a = Math.min(cw / scene.iw, ch / scene.ih);
  fitA = a;
  view = { a, e: M + (cw - scene.iw * a) / 2, f: M + (ch - scene.ih * a) / 2 };
  viewFitDims = { iw: scene.iw, ih: scene.ih, cw: cssW, ch: cssH };
}

// Refit when there is no view yet, or when the ACTUAL drawn image's dimensions
// change (e.g. processing uses the source video, replay uses the smaller cached
// bitmap) or the canvas is resized. Otherwise keep the view so zoom/pan persists.
function ensureFit() {
  if (!view || !viewFitDims ||
      viewFitDims.iw !== scene.iw || viewFitDims.ih !== scene.ih ||
      viewFitDims.cw !== cssW || viewFitDims.ch !== cssH) {
    fitView();
  }
}

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
  const maxX = Math.min(cssW - 2, bounds.x + bounds.w - 2);
  const maxY = Math.min(cssH - 2, bounds.y + bounds.h - 2);
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
  ctx.clearRect(0, 0, canvas.width, canvas.height); // clear the full backing store (device px)
  updateExportUI();
  if (!scene) return;
  ensureFit();
  // One transform maps CSS px -> device px, so every draw below stays in CSS px
  // (view/toCanvasPt/clip/labels all unchanged) yet renders crisp on retina.
  const dpr = dprValue();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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
  // App space is CSS px (same space as toCanvasPt). The canvas display size IS
  // its CSS-px size, so client px map straight in - no canvas.width/rect.width
  // factor (the dpr scaling is handled by the ctx transform in composite()).
  const r = canvas.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
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
    // Pan in CSS px (client-px delta == CSS-px delta); no dpr factor.
    const dx = ev.clientX - pointer.x;
    const dy = ev.clientY - pointer.y;
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
  updateExportUI();
}

// --- export (overlay PNG + kinematics CSV) ---

// The PNG mirrors whatever is currently composited (image, webcam, or the
// selected replay frame). The CSV is the sampled per-frame landmarks behind the
// kinematics panel - a visualization export, not a calibrated measurement.
function updateExportUI() {
  const pngBtn = el("export-png");
  if (pngBtn) pngBtn.hidden = !scene;
  const csvBtn = el("export-csv");
  if (csvBtn) csvBtn.hidden = !(mode === "video" && videoFrames.length > 0);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportPNG() {
  if (!scene) { setStatus("Nothing to export yet.", "error"); return; }
  canvas.toBlob((blob) => {
    if (!blob) { setStatus("Could not export the overlay image.", "error"); return; }
    downloadBlob(blob, "neokine-overlay.png");
    setStatus("Saved overlay PNG.", "ready");
  }, "image/png");
}

function exportCSV() {
  if (mode !== "video" || !videoFrames.length) {
    setStatus("Kinematics CSV is available after processing a video.", "error");
    return;
  }
  const header = ["frame", "t_seconds"];
  for (const name of LANDMARK_NAMES) header.push(name + "_x", name + "_y", name + "_visibility");
  const rows = [header.join(",")];
  videoFrames.forEach((fr, k) => {
    const lm = fr.result.landmarks[0];
    const cells = [k, fr.t.toFixed(4)];
    for (let i = 0; i < LANDMARK_NAMES.length; i++) {
      const p = lm && lm[i];
      if (p) cells.push(p.x.toFixed(6), p.y.toFixed(6), p.visibility != null ? p.visibility.toFixed(4) : "");
      else cells.push("", "", "");
    }
    rows.push(cells.join(","));
  });
  const blob = new Blob([rows.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
  downloadBlob(blob, "neokine-kinematics.csv");
  setStatus("Saved kinematics CSV (" + videoFrames.length + " frames, normalized image units).", "ready");
}

el("export-png").addEventListener("click", exportPNG);
el("export-csv").addEventListener("click", exportCSV);

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

async function handleImageFile(file) {
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
}

el("image-input").addEventListener("change", (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if (!file) return;
  handleImageFile(file);
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
let replayLoop = true;   // loop ON by default
let replayTimer = null;
let replayBitmap = null; // currently decoded frame (closed when replaced)
let replayToken = 0;     // guards out-of-order async decodes during scrubbing
let replayPlayToken = 0; // ensures at most one play loop exists at a time
let videoDirty = false;      // video-mode settings changed since last process
let videoProcessing = false; // a processVideoFile loop is currently running

// In video mode a reprocess is expensive, so settings changes only mark dirty
// and show an indicator; the user reprocesses once via Update.
function markVideoDirty() {
  if (mode !== "video" || !lastVideoFile) return;
  videoDirty = true;
  el("vid-dirty").hidden = false;
}
function clearVideoDirty() {
  videoDirty = false;
  el("vid-dirty").hidden = true;
}
// Enable/disable the reprocess-triggering controls and toggle the Cancel button
// while a video is processing. Kinematics/scrub stay responsive (they do not
// reprocess), and the progress indicator + Cancel remain live.
function setProcessingUI(on) {
  ["#variant button", "#numposes button", "#vid-duration button", "#vid-fps button"]
    .forEach((sel) => document.querySelectorAll(sel).forEach((b) => { b.disabled = on; }));
  el("det").disabled = on;
  el("pres").disabled = on;
  el("apply-btn").disabled = on;
  el("reset-btn").disabled = on;
  el("vid-proc").hidden = !on;
}

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

  // Establish the processing guards SYNCHRONOUSLY, before the first await, so a
  // second Update click during the video-load window is ignored: applyNow guards
  // on videoProcessing, and setProcessingUI(true) disables the controls from the
  // instant Update is clicked (not seconds later, after loadeddata).
  videoProcessing = true;
  setProcessingUI(true);

  videoURL = URL.createObjectURL(file);
  vidfile.src = videoURL;
  setStatus("Loading video...");
  if (vidfile.readyState < 2) await waitEvent(vidfile, "loadeddata");
  if (myRun !== videoRunId) return; // superseded (cancel/new run owns the UI state)
  if (!(await ensureLandmarker())) {
    // Genuine failure (not superseded): release the guards we set above so the
    // UI does not get stuck in the processing state.
    if (myRun === videoRunId) { videoProcessing = false; setProcessingUI(false); }
    return;
  }
  if (myRun !== videoRunId) return;

  setModelState("running", "running");
  const cap = videoDurationSec;               // seconds
  const step = 1 / videoFps;                   // seconds between samples
  const maxFrames = Math.min(VIDEO_MAX_FRAMES, Math.ceil(cap * videoFps) + 1);
  let count = 0;
  let endedEarly = false;
  let t = 0;
  while (t < cap && count < maxFrames) {
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
      const capd = await frameToBlob(vidfile, vidfile.videoWidth, vidfile.videoHeight);
      if (myRun !== videoRunId) return; // cancelled during encode: do not push or overwrite status
      // toBlob can return null under memory pressure; a null blob would later
      // reject in showReplayFrame and wedge finalization, so skip that frame.
      if (capd.blob) videoFrames.push({ t: actual, result: frameResult, blob: capd.blob, w: capd.w, h: capd.h });
    }
    count++;
    const pct = Math.min(100, Math.round((t / cap) * 100));
    setStatus("Processing video... " + actual.toFixed(1) + "s (" + pct + "%, " + count + " frames)");
    await new Promise((r) => setTimeout(r, 0));
    t = Math.round((t + step) * 1000) / 1000;
  }
  if (myRun !== videoRunId) return;

  let longer = false;
  if (!endedEarly) {
    if (isFinite(vidfile.duration) && vidfile.duration > cap + 0.05) longer = true;
    else {
      await seekTo(cap + 0.3);
      if (myRun !== videoRunId) return;
      if (vidfile.currentTime > cap + 0.05) longer = true;
    }
  }
  notice.textContent = longer ? ("Showing first " + cap + " seconds") : "";

  // Frames are cached as blobs; the video element/URL is no longer needed for
  // replay (approach 2). Free it now; resetWorkspace revokes again as a no-op.
  if (videoURL) { URL.revokeObjectURL(videoURL); videoURL = null; }
  vidfile.removeAttribute("src");
  vidfile.load();
  el("rp-notice").textContent = "Sampled playback (~" + videoFps + " fps), not real-time. These are the sampled frames, not the source video.";
  if (videoFrames.length) {
    replayIdx = videoFrames.length - 1;
    showKinPanel(true);   // show the panel FIRST so buildKinJoints -> rebuildKinPlots
    buildKinJoints();     // can build/render the plots (it no-ops while hidden)
    // A frame that fails to decode must NOT leave finalization stuck in the
    // "processing" state (Cancel stuck, status frozen at 99%). Release regardless.
    try { await showReplayFrame(replayIdx); } catch (e) { /* last frame left blank */ }
  } else {
    showKinPanel(false);
  }
  updateReplayUI();

  videoProcessing = false;
  setProcessingUI(false);
  clearVideoDirty();
  setModelState("ready", "ready");
  setStatus("Processed " + (longer ? "first " + cap + " seconds" : "video") + " (" + count + " frames at " + videoFps + " fps).", "ready");
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
  blitAllKinCursors(); // scrub/play: only move the cursor (no recompute, no metrics rebuild)
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
  // replayIdx is briefly -1 while playReplay restarts from the end; clamp so the
  // display never dereferences videoFrames[-1] (which threw and froze playback).
  const idx = Math.max(0, Math.min(M - 1, replayIdx));
  el("replay").hidden = M === 0;
  el("rp-play").textContent = replayPlaying ? "Pause" : "Play";
  const sc = el("rp-scrub");
  sc.max = String(Math.max(0, M - 1));
  sc.value = String(idx);
  el("rp-loop").checked = replayLoop;
  const cur = M ? videoFrames[idx].t : 0;
  const total = M ? videoFrames[M - 1].t : 0;
  el("rp-time").textContent = cur.toFixed(1) + "s / " + total.toFixed(1) + "s";
  el("rp-counter").textContent = M ? "frame " + (idx + 1) + " / " + M : "frame 0 / 0";
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
  el("rp-time").textContent = "0.0s / 0.0s";
  el("rp-play").textContent = "Play";
  clearKinematics();
}

el("rp-play").addEventListener("click", () => { replayPlaying ? pauseReplay() : playReplay(); });
el("rp-back").addEventListener("click", () => { pauseReplay(); showReplayFrame(replayIdx - 1); });
el("rp-fwd").addEventListener("click", () => { pauseReplay(); showReplayFrame(replayIdx + 1); });
el("rp-loop").addEventListener("change", (e) => { replayLoop = e.target.checked; });
el("rp-scrub").addEventListener("input", (e) => { const k = Number(e.target.value); pauseReplay(); showReplayFrame(k); });

// --- sampling knobs (duration + fps) ---

function updateFrameCountNote() {
  const est = Math.min(VIDEO_MAX_FRAMES, Math.ceil(videoDurationSec * videoFps));
  el("vid-frames-note").textContent = "about " + est + " frames (hard cap " + VIDEO_MAX_FRAMES + ")";
  el("vid-heavy-note").hidden = est <= VIDEO_HEAVY_FRAMES;
}
el("vid-duration").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-sec]"); if (!b) return;
  videoDurationSec = Number(b.dataset.sec);
  for (const x of el("vid-duration").querySelectorAll("button")) x.setAttribute("aria-checked", String(x === b));
  updateFrameCountNote();
  markVideoDirty();
});
el("vid-fps").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-fps]"); if (!b) return;
  videoFps = Number(b.dataset.fps);
  for (const x of el("vid-fps").querySelectorAll("button")) x.setAttribute("aria-checked", String(x === b));
  updateFrameCountNote();
  markVideoDirty();
});

// --- kinematics (Part D): computed from the cached per-frame landmarks ---

const KIN_JOINTS = [
  [0, "nose"], [11, "left shoulder"], [12, "right shoulder"], [13, "left elbow"],
  [14, "right elbow"], [15, "left wrist"], [16, "right wrist"], [23, "left hip"],
  [24, "right hip"], [25, "left knee"], [26, "right knee"], [27, "left ankle"], [28, "right ankle"],
];
const KIN_COLORS = ["#2f6df6", "#e8590c", "#0ca678", "#ae3ec9", "#f08c00", "#e64980", "#1098ad", "#5c940d"];
let kinSelected = new Set([15, 16]); // default: both wrists
let kinSmooth = false;

// --- kinematics plot state (display only; the math below is unchanged) ---
// One cached offscreen layer per selected joint (small multiples). Rendering is
// split: renderAllKinPlots() draws each series to its offscreen layer once per
// data/selection/zoom/resize change; blitAllKinCursors() only blits the cached
// layer + the cursor line on every scrub. Instrumentation counters make the
// "scrub redraws cursor only" claim verifiable.
const kinPlots = new Map(); // idx -> { item, name, scroll, canvas, ctx, off, offctx, cssW, cssH, dpr }
let kinZoom = false;        // false = fit-to-width envelope; true = fixed px/sample + horizontal scroll
let kinCollapsed = false;
let kinResizeObs = null;
const KIN_PX_PER_SAMPLE = 6; // zoom mode: pixels per sample in the scroll container

function jointName(idx) { return (KIN_JOINTS.find((j) => j[0] === idx) || [0, "?"])[1]; }
function kinColor(idx) { return KIN_COLORS[[...kinSelected].indexOf(idx) % KIN_COLORS.length]; }

function buildKinJoints() {
  invalidateKinCache(); // fresh data from a new process
  const host = el("kin-joints");
  host.innerHTML = "";
  if (kinSelected.size === 0) kinSelected = new Set([15, 16]);
  for (const [idx, label] of KIN_JOINTS) {
    const l = document.createElement("label");
    l.className = "check small";
    const c = document.createElement("input");
    c.type = "checkbox"; c.checked = kinSelected.has(idx);
    c.addEventListener("change", () => { c.checked ? kinSelected.add(idx) : kinSelected.delete(idx); rebuildKinPlots(); });
    l.appendChild(c);
    const s = document.createElement("span"); s.textContent = label; l.appendChild(s);
    host.appendChild(l);
  }
  rebuildKinPlots(); // populate plots/metrics for the initial selection
}

function smoothInPlace(a) {
  const b = a.slice();
  for (let i = 0; i < a.length; i++) {
    const lo = Math.max(0, i - 1), hi = Math.min(a.length - 1, i + 1);
    let s = 0, n = 0; for (let j = lo; j <= hi; j++) { s += b[j]; n++; }
    a[i] = s / n;
  }
}
// Normalized-image-unit series per joint, cached. The series only depend on the
// cached landmarks + smoothing, so they are computed once and reused across
// scrubs; invalidateKinCache() is called when the data or smoothing changes.
let kinSeriesCache = null;
function invalidateKinCache() { kinSeriesCache = null; }
function kinSeries(idx) {
  if (!kinSeriesCache) kinSeriesCache = new Map();
  const hit = kinSeriesCache.get(idx);
  if (hit) return hit;
  const t = [], x = [], y = [];
  for (const fr of videoFrames) {
    const lm = fr.result.landmarks[0] && fr.result.landmarks[0][idx];
    if (!lm) continue;
    t.push(fr.t); x.push(lm.x); y.push(lm.y);
  }
  if (kinSmooth) { smoothInPlace(x); smoothInPlace(y); }
  const s = { t, x, y };
  kinSeriesCache.set(idx, s);
  return s;
}
function kinPathLength(s) {
  let d = 0;
  for (let i = 1; i < s.x.length; i++) d += Math.hypot(s.x[i] - s.x[i - 1], s.y[i] - s.y[i - 1]);
  return d;
}
function kinMeanVel(s) {
  const dt = s.t.length > 1 ? s.t[s.t.length - 1] - s.t[0] : 0;
  return dt > 0 ? kinPathLength(s) / dt : 0;
}

// The x-axis is the GLOBAL clip time span so every small multiple is aligned
// and shares one cursor. Series times are a subset of the frame times, so
// mapping by the global span is exact.
function kinTimeAxis() {
  const t0 = videoFrames[0].t;
  const total = videoFrames[videoFrames.length - 1].t;
  return { t0, span: Math.max(1e-6, total - t0) };
}

// Draw one value-series (x or y) into an already-DPR-scaled context, in CSS px.
// Fit mode with many samples per pixel would alias if drawn point-per-sample,
// so at >= ~1.5 samples/px we draw a min/max envelope (one vertical bar per
// pixel column, waveform-style); sparser data is drawn as a plain polyline.
// Binning is a DISPLAY choice only - it does not touch the cached series or any
// computed metric.
function drawValueSeries(g, s, values, cssW, cssH, color, dashed) {
  const n = values.length;
  if (n < 2) return;
  const ax = kinTimeAxis();
  const pad = 3;
  const X = (t) => ((t - ax.t0) / ax.span) * (cssW - 1);
  const Y = (v) => pad + (1 - Math.max(0, Math.min(1, v))) * (cssH - 2 * pad);
  g.strokeStyle = color;
  g.lineWidth = 1;
  if (n / cssW >= 1.5) {
    // Envelope: per pixel column, the min..max of the samples falling in it.
    const colMin = new Float32Array(cssW).fill(Infinity);
    const colMax = new Float32Array(cssW).fill(-Infinity);
    for (let i = 0; i < n; i++) {
      let px = Math.round(X(s.t[i]));
      if (px < 0) px = 0; else if (px >= cssW) px = cssW - 1;
      const v = values[i];
      if (v < colMin[px]) colMin[px] = v;
      if (v > colMax[px]) colMax[px] = v;
    }
    g.setLineDash([]);
    g.beginPath();
    for (let px = 0; px < cssW; px++) {
      if (colMax[px] < colMin[px]) continue; // empty column
      g.moveTo(px + 0.5, Y(colMax[px]));
      g.lineTo(px + 0.5, Y(colMin[px]) + 0.001);
    }
    g.stroke();
  } else {
    g.setLineDash(dashed ? [3, 3] : []);
    g.beginPath();
    for (let i = 0; i < n; i++) {
      const px = X(s.t[i]), py = Y(values[i]);
      i ? g.lineTo(px, py) : g.moveTo(px, py);
    }
    g.stroke();
    g.setLineDash([]);
  }
}

function makeKinPlot(idx) {
  const item = document.createElement("div"); item.className = "kin-plot-item";
  const name = document.createElement("div"); name.className = "kin-plot-name";
  const sw = document.createElement("span"); sw.className = "kin-sw"; sw.style.background = kinColor(idx);
  name.appendChild(sw); name.appendChild(document.createTextNode(jointName(idx)));
  const scroll = document.createElement("div"); scroll.className = "kin-plot-scroll";
  // An oversized <canvas> (a replaced element) does not expand its scroll
  // container's scrollWidth on its own, so the horizontal scrollbar never
  // appears in zoom mode. An inner wrapper sized in CSS px fixes that.
  const inner = document.createElement("div"); inner.className = "kin-plot-canvas-wrap";
  const canvas = document.createElement("canvas");
  inner.appendChild(canvas); scroll.appendChild(inner);
  item.appendChild(name); item.appendChild(scroll);
  const off = document.createElement("canvas");
  return { idx, item, name, sw, scroll, inner, canvas, ctx: canvas.getContext("2d"), off, offctx: off.getContext("2d"), cssW: 0, cssH: 56, dpr: 1 };
}

// Retina-correct sizing: back the canvas with cssW*dpr x cssH*dpr device pixels
// and present it at cssW x cssH CSS px, else it renders blurry on dpr > 1.
function sizeKinCanvas(p, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const dw = Math.max(1, Math.round(cssW * dpr));
  const dh = Math.max(1, Math.round(cssH * dpr));
  for (const cv of [p.canvas, p.off]) { cv.width = dw; cv.height = dh; }
  p.canvas.style.width = cssW + "px"; p.canvas.style.height = cssH + "px";
  p.inner.style.width = cssW + "px"; p.inner.style.height = cssH + "px";
  p.cssW = cssW; p.cssH = cssH; p.dpr = dpr;
}

// Render one plot's series to its cached offscreen layer (expensive; once per
// data/selection/zoom/resize change).
function renderKinOffscreen(p) {
  const g = p.offctx;
  if (!p.off.width || !p.off.height) return; // zero-size guard -> no NaN scaling
  g.setTransform(p.dpr, 0, 0, p.dpr, 0, 0);
  g.clearRect(0, 0, p.cssW, p.cssH);
  // baseline
  g.strokeStyle = "rgba(0,0,0,0.10)"; g.lineWidth = 1;
  g.setLineDash([]); g.beginPath();
  g.moveTo(0, p.cssH / 2 + 0.5); g.lineTo(p.cssW, p.cssH / 2 + 0.5); g.stroke();
  const s = kinSeries(p.idx);
  drawValueSeries(g, s, s.x, p.cssW, p.cssH, kinColor(p.idx), false);          // x
  drawValueSeries(g, s, s.y, p.cssW, p.cssH, "rgba(90,100,110,0.85)", true);   // y
}

// Blit the cached layer + the current-frame cursor. Cheap; runs on every scrub.
function blitKinPlot(p) {
  const g = p.ctx;
  if (!p.canvas.width || !p.canvas.height) return; // zero-size guard
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, p.canvas.width, p.canvas.height);
  g.drawImage(p.off, 0, 0); // both device-pixel sized, 1:1
  const ax = kinTimeAxis();
  const cx = ((videoFrames[replayIdx].t - ax.t0) / ax.span) * (p.cssW - 1);
  g.setTransform(p.dpr, 0, 0, p.dpr, 0, 0);
  g.strokeStyle = "rgba(217,122,0,0.95)"; g.lineWidth = 1; g.setLineDash([]);
  g.beginPath(); g.moveTo(cx + 0.5, 0); g.lineTo(cx + 0.5, p.cssH); g.stroke();
  // Zoom mode: keep the cursor within the scroll viewport while scrubbing.
  if (kinZoom) {
    const view = p.scroll.clientWidth;
    if (p.cssW > view) {
      const target = cx - view / 2;
      p.scroll.scrollLeft = Math.max(0, Math.min(p.cssW - view, target));
    }
  }
}

// Scrub path: blit every plot's cached layer + cursor. No series recompute, no
// metrics-DOM rebuild.
function blitAllKinCursors() {
  if (el("kinpanel").hidden || kinCollapsed || !videoFrames.length) return;
  for (const p of kinPlots.values()) blitKinPlot(p);
}

// Size + render + blit every plot. Called on data/selection/zoom/resize/reopen.
function renderAllKinPlots() {
  if (el("kinpanel").hidden || kinCollapsed || !videoFrames.length) return;
  const body = el("kinpanel-body");
  const avail = body.clientWidth - 26; // minus body padding
  if (avail <= 0) return; // hidden/zero-width -> defer (ResizeObserver/reopen re-fires)
  const n = videoFrames.length;
  for (const p of kinPlots.values()) {
    const cssW = kinZoom ? Math.max(avail, Math.round(n * KIN_PX_PER_SAMPLE)) : avail;
    sizeKinCanvas(p, cssW, p.cssH);
    renderKinOffscreen(p);
    blitKinPlot(p);
  }
}

// Reconcile the plot list with the current selection (add/remove/re-color),
// then render. Keeps the swatch colors in sync since colors are position-based.
function reconcileKinPlots() {
  const host = el("kin-plots");
  for (const idx of [...kinPlots.keys()]) {
    if (!kinSelected.has(idx)) { kinPlots.get(idx).item.remove(); kinPlots.delete(idx); }
  }
  for (const idx of kinSelected) {
    if (!kinPlots.has(idx)) { const p = makeKinPlot(idx); kinPlots.set(idx, p); host.appendChild(p.item); }
  }
  for (const p of kinPlots.values()) p.sw.style.background = kinColor(p.idx); // colors are selection-order based
  if (!kinSelected.size) host.innerHTML = "";
}

// Per-joint displacement/velocity + left-right asymmetry. Math unchanged.
function updateKinMetrics() {
  const metrics = el("kin-metrics");
  const sel = [...kinSelected];
  metrics.innerHTML = "";
  sel.forEach((idx) => {
    const s = kinSeries(idx);
    if (s.x.length < 2) return;
    const row = document.createElement("div"); row.className = "kin-row";
    row.innerHTML = '<span><span class="kin-sw" style="background:' + kinColor(idx) + '"></span>' + jointName(idx) + "</span>" +
      "<span>disp " + kinPathLength(s).toFixed(3) + " | vel " + kinMeanVel(s).toFixed(3) + "/s</span>";
    metrics.appendChild(row);
  });
  if (!sel.length) metrics.innerHTML = '<div class="kin-row">select one or more joints above</div>';

  const pair = el("kin-pair").value.split(",").map(Number);
  const L = kinPathLength(kinSeries(pair[0])), R = kinPathLength(kinSeries(pair[1]));
  const denom = L + R;
  const ai = denom > 0 ? (L - R) / denom : 0;
  const side = Math.abs(ai) < 0.02 ? "symmetric" : (ai > 0 ? "left moved more" : "right moved more");
  el("kin-asym").textContent = denom > 0 ? ai.toFixed(2) + " (" + side + ")" : "-";
}

// Full refresh: reconcile plots, render layers, update metrics. Called on
// selection/smoothing/data/zoom change - NOT on scrub.
function rebuildKinPlots() {
  if (el("kinpanel").hidden || !videoFrames.length) return;
  reconcileKinPlots();
  renderAllKinPlots();
  updateKinMetrics();
}

// Show/hide + expand/collapse the right-side panel. data-kin drives the grid
// column width and the collapsed presentation (see styles.css).
function showKinPanel(hasData) {
  const panel = el("kinpanel");
  if (!hasData) {
    panel.hidden = true;
    document.body.dataset.kin = "off";
    return;
  }
  panel.hidden = false;
  document.body.dataset.kin = kinCollapsed ? "collapsed" : "open";
}

function toggleKinPanel() {
  kinCollapsed = !kinCollapsed;
  document.body.dataset.kin = kinCollapsed ? "collapsed" : "open";
  el("kin-toggle").textContent = kinCollapsed ? "Expand" : "Collapse";
  // Expanding restores a non-zero body width; re-render the layers (they may
  // have been sized against a 0-width body while collapsed).
  if (!kinCollapsed) renderAllKinPlots();
}

function clearKinematics() {
  invalidateKinCache();
  kinSelected.clear();
  kinSmooth = false;
  if (el("kin-smooth")) el("kin-smooth").checked = false;
  if (el("kin-smooth-note")) el("kin-smooth-note").hidden = true;
  if (el("kin-joints")) el("kin-joints").innerHTML = "";
  if (el("kin-metrics")) el("kin-metrics").innerHTML = "";
  if (el("kin-asym")) el("kin-asym").textContent = "-";
  for (const p of kinPlots.values()) p.item.remove();
  kinPlots.clear();
  if (el("kin-plots")) el("kin-plots").innerHTML = "";
  showKinPanel(false);
}

el("kin-smooth").addEventListener("change", (e) => {
  kinSmooth = e.target.checked;
  el("kin-smooth-note").hidden = !kinSmooth;
  invalidateKinCache(); // smoothing changes the series
  rebuildKinPlots();
});
el("kin-pair").addEventListener("change", updateKinMetrics);
el("kin-zoom").addEventListener("change", (e) => {
  kinZoom = e.target.checked;
  renderAllKinPlots(); // display-only: re-render layers at the new width
});
el("kin-toggle").addEventListener("click", toggleKinPanel);

// Re-render the cached layers when the panel width changes (rail scroll, window
// resize, responsive stack). Guarded against zero-size inside renderAllKinPlots.
if (window.ResizeObserver) {
  kinResizeObs = new ResizeObserver(() => renderAllKinPlots());
  kinResizeObs.observe(el("kinpanel-body"));
}

function handleVideoFile(file) {
  if (file.size > VIDEO_MAX_BYTES) {
    cancelVideo();
    clearRenderState();
    el("video-name").textContent = file.name;
    setStatus("File too large (" + (file.size / 1048576).toFixed(0) + " MB). Limit is " + Math.round(VIDEO_MAX_BYTES / 1048576) + " MB.", "error");
    return;
  }
  clearVideoDirty();
  processVideoFile(file);
}

el("video-input").addEventListener("change", (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if (!file) return;
  handleVideoFile(file);
});

// Bundled demo clip: fetched from docs/samples/ and fed through the exact same
// path as an uploaded file (nothing is uploaded). It is AI-generated footage of
// a synthetic person - see docs/samples/README.md.
async function loadSampleVideo() {
  const btn = el("sample-video");
  btn.disabled = true;
  setStatus("Loading sample clip...");
  try {
    const res = await fetch("samples/synthetic-walk.mp4");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const blob = await res.blob();
    const file = new File([blob], "synthetic-walk.mp4", { type: blob.type || "video/mp4" });
    if (mode !== "video") switchMode("video");
    handleVideoFile(file);
  } catch (e) {
    setStatus("Could not load the sample clip. " + e, "error");
  } finally {
    btn.disabled = false;
  }
}
el("sample-video").addEventListener("click", loadSampleVideo);

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
  viewFitDims = null;
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
  videoProcessing = false;
  setProcessingUI(false);
  clearVideoDirty();
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

// --- drag & drop onto the stage ---
// Dropping an image or a video loads it through the same path as the file
// pickers, auto-switching to the matching mode. preventDefault on dragover is
// what tells the browser this is a drop target (otherwise it navigates).
for (const evName of ["dragenter", "dragover"]) {
  stage.addEventListener(evName, (e) => {
    if (!e.dataTransfer || Array.from(e.dataTransfer.types || []).indexOf("Files") === -1) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    stage.classList.add("dragover");
  });
}
for (const evName of ["dragleave", "dragend"]) {
  stage.addEventListener(evName, (e) => {
    // Ignore dragleave that just crossed into a child element still inside the stage.
    if (evName === "dragleave" && e.relatedTarget && stage.contains(e.relatedTarget)) return;
    stage.classList.remove("dragover");
  });
}
stage.addEventListener("drop", (e) => {
  e.preventDefault();
  stage.classList.remove("dragover");
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (!file) return;
  if (file.type.startsWith("image/")) {
    if (mode !== "image") switchMode("image");
    handleImageFile(file);
  } else if (file.type.startsWith("video/")) {
    if (mode !== "video") switchMode("video");
    handleVideoFile(file);
  } else {
    setStatus("Unsupported file. Drop an image or a video.", "error");
  }
});

// --- model + settings controls ---

el("variant").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-variant]");
  if (!btn || btn.dataset.variant === variantByMode[mode]) return;
  variantByMode[mode] = btn.dataset.variant;
  updateVariantUI();
  dirty = true;
  if (mode === "image") { const ok = await ensureLandmarker(); if (ok && lastImage) detectImage(); }
  else if (mode === "video") markVideoDirty(); // reprocess is expensive - wait for Update
});

// Auto-rerun is only for IMAGE mode (cheap). Video mode uses markVideoDirty +
// Update; webcam applies on the next frame.
let rerunTimer = null;
function scheduleRerun() {
  clearTimeout(rerunTimer);
  rerunTimer = setTimeout(() => { if (mode === "image" && lastImage) detectImage(); }, 250);
}

el("numposes").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-step]");
  if (!btn) return;
  settings.numPoses = Math.max(1, Math.min(4, settings.numPoses + Number(btn.dataset.step)));
  el("numposes-val").textContent = String(settings.numPoses);
  dirty = true;
  if (mode === "video") markVideoDirty(); else scheduleRerun();
});

function bindSlider(id, valId, key) {
  const slider = el(id), out = el(valId);
  slider.addEventListener("input", () => {
    settings[key] = Number(slider.value);
    out.textContent = settings[key].toFixed(2);
    dirty = true;
    if (mode === "video") markVideoDirty(); else scheduleRerun();
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
  else if (mode === "video" && lastVideoFile) {
    if (videoProcessing) return; // one reprocess at a time; ignore while running
    clearVideoDirty();
    processVideoFile(lastVideoFile);
  }
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
  else if (mode === "video" && lastVideoFile) markVideoDirty(); // reprocess is expensive - wait for Update
  else composite();
}
el("apply-btn").addEventListener("click", applyNow);
el("reset-btn").addEventListener("click", resetSettings);
el("vid-cancel").addEventListener("click", cancelProcessing);

// Cancel an in-progress reprocess: stop the loop, revoke the URL, clean state.
function cancelProcessing() {
  videoRunId++;                 // the running loop's myRun !== videoRunId -> it bails
  if (videoURL) { URL.revokeObjectURL(videoURL); videoURL = null; }
  vidfile.removeAttribute("src"); vidfile.load();
  videoProcessing = false;
  setProcessingUI(false);
  clearRenderState();           // drop any partially-cached frames / replay / kinematics
  markVideoDirty();             // a file is loaded but not processed - offer Update
  setModelState("ready", "ready");
  setStatus("Processing cancelled.", "ready");
}

window.addEventListener("pagehide", () => { teardownCamera(); cancelVideo(); });

const ro = new ResizeObserver(() => { sizeCanvas(); if (scene) { fitView(); composite(); } });
ro.observe(stage);

// Re-crisp the backing store when devicePixelRatio changes (e.g. dragging the
// window between a retina and a non-retina monitor). The CSS size may not change
// then, so the ResizeObserver above won't fire; a matchMedia resolution listener
// catches it. Re-registered each time because the query embeds the current dpr.
function watchDprChange() {
  const mq = window.matchMedia("(resolution: " + window.devicePixelRatio + "dppx)");
  const onChange = () => {
    sizeCanvas();
    if (scene) { fitView(); composite(); }
    watchDprChange(); // re-arm for the new dpr
  };
  if (mq.addEventListener) mq.addEventListener("change", onChange, { once: true });
  else mq.addListener(onChange); // legacy Safari
}
watchDprChange();

// --- logo click: full reset to first-load state (in-page, no navigation) ---
// resetWorkspace() is the real teardown (stops webcam, cancels in-flight video
// processing, clears the loaded input/canvas/replay/kinematics); then restore
// default mode/settings and the expanded sections, exactly as on first load.
function resetApp() {
  resetWorkspace();
  mode = "image";
  document.body.dataset.mode = mode;
  for (const b of document.querySelectorAll(".mode-btn[data-mode]")) b.classList.toggle("is-active", b.dataset.mode === mode);
  for (const p of document.querySelectorAll(".mode-panel")) p.hidden = p.dataset.for !== mode;
  placeholderText.textContent = PLACEHOLDER.image;
  const adv = document.querySelector("details.advanced"); if (adv) adv.open = true; // sections expanded as on load
  resetSettings(); // default variant/sliders/Face-off/interactions (composites the now-empty scene)
  setModelState("ready", "ready");
  setStatus(MODE_STATUS.image, "ready");
}
el("brand-reset").addEventListener("click", (e) => { e.preventDefault(); resetApp(); });

// --- (i) info popovers: tap/click/keyboard to toggle, hover to preview on
// desktop. Rendered position:fixed and appended to <body> so the rail's
// overflow scroll never clips it. Escape and outside-click dismiss. ---
(function initInfoPopovers() {
  const pop = document.createElement("div");
  pop.className = "infopop"; pop.id = "infopop"; pop.setAttribute("role", "tooltip"); pop.hidden = true;
  document.body.appendChild(pop);
  let cur = null, sticky = false;

  function place(btn) {
    const r = btn.getBoundingClientRect();
    const pad = 8, pw = pop.offsetWidth, ph = pop.offsetHeight;
    let left = r.right + pad;
    if (left + pw > window.innerWidth - pad) left = r.left - pw - pad;   // flip to the left
    left = Math.max(pad, Math.min(left, window.innerWidth - pw - pad));
    let top = Math.min(r.top, window.innerHeight - ph - pad);
    top = Math.max(pad, top);
    pop.style.left = Math.round(left) + "px";
    pop.style.top = Math.round(top) + "px";
  }
  function show(btn, isSticky) {
    if (cur && cur !== btn) hide();
    pop.textContent = btn.getAttribute("data-info") || "";
    pop.hidden = false;
    cur = btn; sticky = isSticky;
    btn.setAttribute("aria-expanded", "true");
    btn.setAttribute("aria-describedby", "infopop");
    place(btn);
  }
  function hide() {
    if (!cur) return;
    cur.setAttribute("aria-expanded", "false");
    cur.removeAttribute("aria-describedby");
    cur = null; sticky = false; pop.hidden = true;
  }
  // click = tap = keyboard (buttons fire click on Enter/Space). preventDefault so
  // an (i) inside a <summary> does not also toggle the <details>.
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".info");
    if (btn) { e.preventDefault(); e.stopPropagation(); (cur === btn && sticky) ? hide() : show(btn, true); return; }
    if (cur && !pop.contains(e.target)) hide();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && cur) { const b = cur; hide(); b.focus(); } });
  // Optional desktop hover preview (tap remains the primary path; skip on touch).
  document.addEventListener("pointerover", (e) => {
    if (e.pointerType === "touch") return;
    const btn = e.target.closest(".info");
    if (btn && !sticky) show(btn, false);
  });
  document.addEventListener("pointerout", (e) => {
    if (e.pointerType === "touch") return;
    const btn = e.target.closest(".info");
    if (btn && cur === btn && !sticky) hide();
  });
  window.addEventListener("resize", hide);
  const reflow = () => { if (cur) place(cur); };
  document.querySelector(".rail") && document.querySelector(".rail").addEventListener("scroll", reflow);
  const kb = document.getElementById("kinpanel-body"); if (kb) kb.addEventListener("scroll", reflow);
})();

// initial load
sizeCanvas();
buildFilters();
updateVariantUI();
updateInteractiveUI();
updateFrameCountNote();
setStatus("Loading pose model...");
ensureLandmarker().then((ok) => { if (ok) setStatus("Pose model ready. Choose an image.", "ready"); });
