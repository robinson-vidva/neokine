// Enhanced inference — the SyRIP-tuned pipeline, ported from the Python
// analysis project's core/pipeline.py. It wraps a plain PoseLandmarker and
// adds the inference improvements that lifted detection 94.6% -> 99.2% on the
// SyRIP held-out test set with zero lost detections:
//
//   * upscale small images (short side -> 512) so tiny infants keep detail,
//   * rotation test-time augmentation: try 0/90/180/270 degrees and keep the
//     orientation MediaPipe is most confident about (mean landmark
//     visibility), which recovers the inverted / sideways / crawling poses
//     the detector otherwise misses entirely.
//
// This is the SAME model — not retrained weights. It returns a result in the
// exact shape PoseLandmarker.detect() returns ({ landmarks: [[{x,y,z,
// visibility}...]] }) plus a `meta` field, so app.js can treat both paths
// identically.

const DEFAULTS = { upscaleMinSide: 512, angles: [0, 90, 180, 270] };

function sourceSize(src) {
  const w = src.naturalWidth || src.videoWidth || src.width;
  const h = src.naturalHeight || src.videoHeight || src.height;
  return [w, h];
}

// Draw `src` rotated clockwise by `deg` (and scaled) into a fresh canvas.
// Returns { canvas, sw, sh, rad } where sw/sh are the scaled source dims.
function rotatedCanvas(src, deg, scale) {
  const [w, h] = sourceSize(src);
  const sw = w * scale, sh = h * scale;
  const rad = (deg * Math.PI) / 180;
  const swap = deg % 180 !== 0;
  const cw = Math.round(swap ? sh : sw);
  const ch = Math.round(swap ? sw : sh);
  const cv = document.createElement("canvas");
  cv.width = cw; cv.height = ch;
  const c = cv.getContext("2d");
  c.translate(cw / 2, ch / 2);
  c.rotate(rad);
  c.drawImage(src, -sw / 2, -sh / 2, sw, sh);
  return { canvas: cv, sw, sh, rad };
}

// Map a normalized point (u,v) in the rotated canvas back to a normalized
// point in the ORIGINAL source frame. This inverts the exact affine used to
// draw (translate to canvas centre, rotate, draw source centred), so it is
// correct for any angle by construction — no per-angle special-casing.
function makeInverse(cw, ch, sw, sh, rad) {
  const cx = cw / 2, cy = ch / 2, sx = sw / 2, sy = sh / 2;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  return (u, v) => {
    const px = u * cw - cx, py = v * ch - cy;
    // inverse rotation R(-rad) applied to (px,py), then back to source frame
    const sxp = cos * px + sin * py + sx;
    const syp = -sin * px + cos * py + sy;
    return [sxp / sw, syp / sh];
  };
}

function meanVisibility(landmarks) {
  if (!landmarks || !landmarks.length) return 0;
  let s = 0, n = 0;
  for (const l of landmarks) { if (typeof l.visibility === "number") { s += l.visibility; n++; } }
  return n ? s / n : 0;
}

// detectEnhanced(landmarker, source, opts) -> { landmarks, worldLandmarks, meta }
// `landmarker` must be an IMAGE-mode PoseLandmarker (uses .detect()).
export function detectEnhanced(landmarker, source, opts = {}) {
  const { upscaleMinSide, angles } = { ...DEFAULTS, ...opts };
  const [w, h] = sourceSize(source);
  const scale = Math.max(1, upscaleMinSide / Math.min(w, h));

  let best = null; // { vis, landmarks, angle }
  let tried = 0;
  for (const deg of angles) {
    const { canvas, sw, sh, rad } = rotatedCanvas(source, deg, scale);
    let res;
    try { res = landmarker.detect(canvas); } catch (e) { continue; }
    tried++;
    if (!res || !res.landmarks || !res.landmarks.length) continue;
    const inv = makeInverse(canvas.width, canvas.height, sw, sh, rad);
    // map every pose's landmarks back into the original frame
    const mapped = res.landmarks.map((pose) =>
      pose.map((l) => {
        const [x, y] = inv(l.x, l.y);
        return { x, y, z: l.z, visibility: l.visibility };
      })
    );
    const vis = meanVisibility(mapped[0]);
    if (!best || vis > best.vis) best = { vis, landmarks: mapped, angle: deg, upscaled: scale > 1 };
  }

  if (!best) return { landmarks: [], worldLandmarks: [], meta: { detected: false, tried } };
  return {
    landmarks: best.landmarks,
    worldLandmarks: [],
    meta: { detected: true, angle: best.angle, upscaled: best.upscaled, visibility: best.vis, tried },
  };
}
