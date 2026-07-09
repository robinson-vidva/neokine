"""neokine - Streamlit pose-visualization app (local, Python/MediaPipe backend).

Feature parity with the web tool (docs/): image / video / webcam input, the
lite/full/heavy variants, numPoses and confidence thresholds, region-colored
skeleton overlay with display filters and labels, interactive zoom/pan + hover
+ click-to-pin (via Plotly), and the kinematics panel for video.

Run from the repo root:  streamlit run app/streamlit_app.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import pose_backend as pb          # noqa: E402
from app import kinematics as kin            # noqa: E402
from app.overlay import build_figure, draw_overlay_rgb, DISPLAY_NAMES  # noqa: E402
from app.viz_config import GROUPS, ASYM_PAIRS         # noqa: E402

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

st.set_page_config(page_title="neokine (Streamlit)", layout="wide")

# Reclaim vertical space above the visualization: shrink the top padding and
# the (empty) Streamlit header bar so the image sits near the top.
st.markdown(
    "<style>"
    "[data-testid='stMainBlockContainer']{padding-top:1.2rem;padding-bottom:1rem;}"
    ".block-container{padding-top:1.2rem;padding-bottom:1rem;}"
    "[data-testid='stHeader']{height:0.5rem;background:transparent;}"
    "</style>",
    unsafe_allow_html=True,
)

CAVEAT = ("Visualization only - normalized image units, sampled, not a "
          "measurement. No calibration, depth, or scale reference; the values "
          "cannot be converted to physical units.")


@st.cache_resource(show_spinner="Loading pose model...")
def get_landmarker(variant, num_poses, min_det, min_pres, mode):
    return pb.make_landmarker(variant, num_poses, min_det, min_pres, mode)


def set_all_joints(val):
    for i in range(33):
        st.session_state["j_%d" % i] = val


def set_group_joints(members, group_key):
    val = st.session_state[group_key]
    for i in members:
        st.session_state["j_%d" % i] = val


def step_frame(delta, n):
    st.session_state.frame_pos = max(0, min(n - 1, st.session_state.get("frame_pos", 0) + delta))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("neokine")
st.sidebar.caption("Neonatal Kinematics - local Python demo")

mode = st.sidebar.radio("Input", ["Upload image", "Upload video", "Live webcam"])

st.sidebar.subheader("Model")
variant = st.sidebar.radio(
    "Variant", ["lite", "full", "heavy"], index=1, horizontal=True,
    help="Model size vs accuracy. lite is fastest, heavy most accurate. "
         "Changing it re-runs detection.")
num_poses = st.sidebar.number_input(
    "numPoses", 1, 4, 1, 1, help="Maximum number of people to detect in a frame.")
min_det = st.sidebar.slider(
    "minPoseDetectionConfidence", 0.0, 1.0, 0.50, 0.05,
    help="Minimum confidence to accept a detection. Lower is not better - it "
         "just lets weaker (less certain) detections through.")
min_pres = st.sidebar.slider(
    "minPosePresenceConfidence", 0.0, 1.0, 0.50, 0.05,
    help="Minimum confidence that a detected pose is actually present.")

st.sidebar.subheader("Overlay")
labels_mode = "always" if st.sidebar.radio(
    "Point labels", ["On hover / selected", "Always show"],
    help="'Always show' draws a text label at each named joint; otherwise "
         "labels appear on hover or when a joint is pinned.") == "Always show" else "hover"
hide_oof = st.sidebar.checkbox(
    "Hide out-of-frame points", value=True,
    help="When a joint is outside the frame MediaPipe still outputs an "
         "estimated position - a guess for a point it cannot see, not an "
         "observation. Hidden by default.")

st.sidebar.caption("Points shown (display filter)")
mc1, mc2 = st.sidebar.columns(2)
mc1.button("Show all", use_container_width=True, on_click=set_all_joints, args=(True,),
           help="Show every landmark.")
mc2.button("Hide all", use_container_width=True, on_click=set_all_joints, args=(False,),
           help="Hide every landmark.")
shown = set()
for gkey, g in GROUPS.items():
    grp_key = "grp_%s" % gkey
    # Keep the group toggle in sync with its joints (checked = all joints on).
    st.session_state[grp_key] = all(
        st.session_state.get("j_%d" % i, g["on"]) for i in g["members"])
    with st.sidebar.expander(g["label"], expanded=False):
        st.checkbox("**%s** (all)" % g["label"], key=grp_key,
                    on_change=set_group_joints, args=(g["members"], grp_key))
        for i in g["members"]:
            if st.checkbox(DISPLAY_NAMES[i], value=g["on"], key="j_%d" % i):
                shown.add(i)

if "pinned" not in st.session_state:
    st.session_state.pinned = set()


def render_overlay(rgb, poses, key, height=780):
    """Draw the Plotly overlay and reconcile click-to-pin selection."""
    fig = build_figure(rgb, poses, shown, labels_mode, hide_oof,
                       frozenset(st.session_state.pinned), height=height)
    event = st.plotly_chart(
        fig, use_container_width=True, key=key,
        on_select="rerun", selection_mode="points",
        config={"scrollZoom": True, "displayModeBar": True},
    )
    new_pin = set()
    sel = (event or {}).get("selection") or {}
    for pt in sel.get("points", []):
        cd = pt.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is not None:
            new_pin.add(int(cd))
    if new_pin != st.session_state.pinned:
        st.session_state.pinned = new_pin
        st.rerun()


def decode_image(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def cleanup_video_proc():
    """Release the in-progress VideoCapture and delete its temp file."""
    cap = st.session_state.pop("vp_cap", None)
    if cap is not None:
        cap.release()
    tmp = st.session_state.pop("vp_tmp", None)
    if tmp and os.path.exists(tmp):
        try:
            os.unlink(tmp)
        except OSError:
            pass


def kinematics_panel(frames):
    st.subheader("Kinematics")
    st.caption(CAVEAT)
    default = [i for i in (15, 16, 27, 28) if any(f["poses"] for f in frames)]
    names = {i: DISPLAY_NAMES[i] for i in range(33)}
    picked = st.multiselect("Track joints", options=list(range(33)),
                            default=default, format_func=lambda i: names[i])
    smooth_on = st.checkbox(
        "Smooth (3-frame average) before velocity", value=False,
        help="Applies a 3-frame moving average to x and y before computing "
             "displacement/velocity. Reduces jitter but changes the estimate.")

    import plotly.graph_objects as go
    rows = []
    if picked:
        fig = go.Figure()
        for i in picked:
            t, x, y = kin.series(frames, i, smooth_on)
            if len(t) < 2:
                continue
            fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name=names[i] + " x"))
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=names[i] + " y",
                                     line=dict(dash="dot"), opacity=0.5))
            rows.append({"joint": names[i],
                         "displacement": round(kin.path_length(x, y), 4),
                         "mean velocity (u/s)": round(kin.mean_velocity(t, x, y), 4)})
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="time (s)", yaxis_title="normalized position",
                          legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rows, use_container_width=True, hide_index=True)

    pair_name = st.selectbox("Left-right asymmetry", list(ASYM_PAIRS.keys()), index=0)
    li, ri = ASYM_PAIRS[pair_name]
    aidx, L, R = kin.asymmetry(frames, li, ri, smooth_on)
    st.metric("Asymmetry index (L-R)/(L+R)", "%+.3f" % aidx,
              help="Range -1..1; 0 = symmetric, + = left moved more. "
                   "L=%.3f R=%.3f (total displacement)." % (L, R))


# ------------------------------------------------------------------ header
st.caption("MediaPipe Pose runs locally in Python. Visualization only - "
           "not a measurement or diagnostic tool; no clinical claims.")

# ------------------------------------------------------------------ IMAGE
if mode == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        rgb = decode_image(up.read())
        lm = get_landmarker(variant, num_poses, min_det, min_pres, "image")
        poses = pb.detect_image(lm, rgb)
        if poses:
            st.success("Pose detected. poses: %d" % len(poses))
        else:
            st.warning("No pose detected.")
        render_overlay(rgb, poses, key="img")
        if st.session_state.pinned:
            st.caption("Pinned: " + ", ".join(DISPLAY_NAMES[i] for i in sorted(st.session_state.pinned)))
    else:
        st.info("Upload an image to run pose.")

# ------------------------------------------------------------------ WEBCAM
elif mode == "Live webcam":
    st.caption("Live pose runs on your webcam. Frames are processed in memory "
               "by the local Python backend and discarded - nothing is recorded.")
    if _HAS_WEBRTC:
        # Capture current settings for the processor (applied when the stream starts).
        _v, _n, _md, _mp = variant, num_poses, min_det, min_pres
        _shown, _hide = set(shown), hide_oof

        class PoseProcessor(VideoProcessorBase):
            def __init__(self):
                self.lm = pb.make_landmarker(_v, _n, _md, _mp, "image")

            def recv(self, frame):
                rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                small = cv2.resize(rgb, (480, int(h * 480 / w))) if w > 480 else rgb
                try:
                    poses = pb.detect_image(self.lm, small)   # normalized coords
                    rgb = draw_overlay_rgb(rgb, poses, _shown, _hide)
                except Exception:
                    pass
                return av.VideoFrame.from_ndarray(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), format="bgr24")

        webrtc_streamer(
            key="live",
            video_processor_factory=PoseProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        st.caption("Model, filters and confidence apply when the stream starts - "
                   "change them, then Stop and Start again.")
    else:
        st.info("Live webcam needs the streamlit-webrtc and av packages "
                "(installing). Using snapshot mode meanwhile.")
        snap = st.camera_input("Take a snapshot")
        if snap:
            rgb = decode_image(snap.read())
            lm = get_landmarker(variant, num_poses, min_det, min_pres, "image")
            poses = pb.detect_image(lm, rgb)
            render_overlay(rgb, poses, key="cam")

# ------------------------------------------------------------------ VIDEO
else:
    up = st.file_uploader("Choose a video", type=["mp4", "mov", "webm", "avi", "mkv"])
    c1, c2 = st.columns(2)
    dur_cap = c1.selectbox("Duration cap (s)", [5, 10, 15, 30], index=1)
    fps = c2.selectbox("Sampling rate (fps)", [5, 10, 15, 30], index=1)

    if up:
        sig = (up.name, up.size, variant, num_poses, min_det, min_pres, dur_cap, fps)

        # (Re)start processing whenever the video or settings change.
        if st.session_state.get("vp_sig") != sig:
            cleanup_video_proc()
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as tf:
                tf.write(up.read())
                tmp_path = tf.name
            cap = cv2.VideoCapture(tmp_path)
            st.session_state.vp_sig = sig
            st.session_state.vp_cap = cap
            st.session_state.vp_tmp = tmp_path
            st.session_state.vp_lm = pb.make_landmarker(variant, num_poses, min_det, min_pres, "video")
            st.session_state.vp_src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            st.session_state.vp_step = max(1, int(round(st.session_state.vp_src_fps / fps)))
            st.session_state.vp_max = int(dur_cap * fps)
            st.session_state.vp_i = 0
            st.session_state.vp_frames = []
            st.session_state.vp_active = True
            st.session_state.vid_frames = []
            st.session_state.vid_sig = None
            st.session_state.pinned = set()
            st.session_state.frame_pos = 0

        # Batched, cancellable processing. A fragment processes a few frames at
        # a time and reruns itself, so the Cancel button stays responsive and the
        # rest of the page doesn't flicker.
        if st.session_state.get("vp_active"):
            mx = st.session_state.vp_max
            # Always-visible status (outer script) so there is a processing
            # indicator even before the first fragment tick renders.
            st.markdown("### ⏳ Processing video…")

            # Progress bar AND Cancel button both live INSIDE the fragment, so
            # they render together in the fragment's own delta path and update
            # every tick (a fragment can't reliably draw to outer elements).
            @st.fragment(run_every=0.05)
            def process_batch():
                if not st.session_state.get("vp_active"):
                    return
                cap = st.session_state.vp_cap
                lm = st.session_state.vp_lm
                step = st.session_state.vp_step
                src_fps = st.session_state.vp_src_fps
                done = False
                for _ in range(5):                   # ~5 source frames per tick
                    if len(st.session_state.vp_frames) >= mx:
                        done = True
                        break
                    ok, bgr = cap.read()
                    if not ok:
                        done = True
                        break
                    if st.session_state.vp_i % step == 0:
                        t = st.session_state.vp_i / src_fps
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        h, w = rgb.shape[:2]
                        scale = 900.0 / max(w, h)
                        if scale < 1:
                            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
                        poses = pb.detect_video_frame(lm, rgb, int(t * 1000))
                        st.session_state.vp_frames.append({"t": t, "rgb": rgb, "poses": poses})
                    st.session_state.vp_i += 1

                kept = len(st.session_state.vp_frames)
                st.progress(min(1.0, kept / mx),
                            text="Processed %d / %d frames" % (kept, mx))
                if st.button("Cancel", type="primary", key="vp_cancel"):
                    st.session_state.vp_active = False
                    cleanup_video_proc()
                    st.rerun()
                if done:
                    st.session_state.vid_frames = st.session_state.vp_frames
                    st.session_state.vid_sig = sig
                    st.session_state.vp_active = False
                    cleanup_video_proc()
                    st.rerun()

            process_batch()
            st.stop()

        frames = st.session_state.get("vid_frames") or []
        if not frames:
            st.info("Processing cancelled. Change a setting or re-upload to process again.")
        else:
            detected = sum(1 for f in frames if f["poses"])
            st.success("Processed %d frames (%d with a pose) at %d fps."
                       % (len(frames), detected, fps))
            n = len(frames)
            st.session_state.setdefault("frame_pos", 0)
            st.session_state.frame_pos = min(st.session_state.frame_pos, n - 1)

            row = st.columns([1, 1, 1, 1, 1])
            playing = st.session_state.get("vid_playing", False)
            if row[0].button("⏹ Stop" if playing else "▶ Play",
                             use_container_width=True,
                             type="primary" if playing else "secondary"):
                st.session_state.vid_playing = not playing
                st.rerun()   # re-render so the label/run_every reflect the new state
            row[1].button("Prev", use_container_width=True, disabled=playing,
                          on_click=step_frame, args=(-1, n), help="Previous frame")
            row[2].button("Next", use_container_width=True, disabled=playing,
                          on_click=step_frame, args=(1, n), help="Next frame")
            row[3].checkbox("Loop", value=True, key="vid_loop")
            speed = row[4].selectbox("Speed", [0.25, 0.5, 1.0, 2.0], index=2,
                                     format_func=lambda x: "%gx" % x,
                                     label_visibility="collapsed",
                                     help="Playback speed multiplier.")

            @st.fragment(run_every=(1.0 / fps / speed) if playing else None)
            def player():
                m = len(st.session_state.vid_frames)
                if st.session_state.get("vid_playing"):
                    nxt = st.session_state.frame_pos + 1
                    if nxt >= m:
                        nxt = 0 if st.session_state.get("vid_loop") else m - 1
                    st.session_state.frame_pos = nxt
                idx = min(st.session_state.frame_pos, m - 1)
                # Seek slider tracks playback (moves each tick); draggable when stopped.
                seek = st.slider("Seek", 0, m - 1, idx, label_visibility="collapsed",
                                 disabled=st.session_state.get("vid_playing", False))
                if not st.session_state.get("vid_playing") and seek != idx:
                    st.session_state.frame_pos = idx = seek
                fr = st.session_state.vid_frames[idx]
                st.caption("t = %.2fs   frame %d / %d" % (fr["t"], idx + 1, m))
                if st.session_state.get("vid_playing"):
                    # Fast image path (no flicker). Styling matches the Plotly
                    # overlay so the skeleton looks the same playing and paused.
                    st.image(draw_overlay_rgb(fr["rgb"], fr["poses"], shown, hide_oof),
                             use_container_width=True)
                else:
                    render_overlay(fr["rgb"], fr["poses"], key="vid", height=720)

            player()
            st.divider()
            kinematics_panel(frames)
    else:
        st.info("Upload a short video to run pose over time.")
