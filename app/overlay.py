"""Build the interactive Plotly overlay: background image + colored skeleton +
joint markers + labels. Plotly gives us the web tool's interactions for free:
native zoom/pan, hover tooltips (joint name + coords + visibility), legend
toggles, and click-to-pin (via st.plotly_chart selection).
"""

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from app.viz_config import (
    REGION_COLORS, CONNECTIONS, GROUP_OF, MAJOR, LABELABLE, joint_color,
)
from shared.skeleton import MEDIAPIPE_LANDMARK_NAMES

DISPLAY_NAMES = [n.replace("_", " ") for n in MEDIAPIPE_LANDMARK_NAMES]


def _hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _in_frame(p, i):
    return 0.0 <= p[i, 0] <= 1.0 and 0.0 <= p[i, 1] <= 1.0


def head_circle(p, visible, w, h):
    """Head as a circle in pixel coords: (cx, cy, r) or None.

    The MediaPipe face landmarks are too sparse to connect meaningfully, so we
    represent the head with a circle sized from the ears (fallback: eyes).
    Gated by `visible` so it respects the Face display filter / out-of-frame.
    """
    def pt(i):
        return np.array([p[i, 0] * w, p[i, 1] * h])

    if visible(7) and visible(8):            # both ears
        c = (pt(7) + pt(8)) / 2
        r = np.linalg.norm(pt(7) - pt(8)) * 0.62
    elif visible(2) and visible(5):          # both eyes
        c = (pt(2) + pt(5)) / 2
        r = np.linalg.norm(pt(2) - pt(5)) * 1.5
    else:
        return None
    return float(c[0]), float(c[1]), float(r)


def draw_overlay_rgb(rgb, poses, shown, hide_oof=True):
    """Fast OpenCV skeleton draw onto a copy of an RGB frame (for playback).

    Cheaper than a Plotly figure, so it animates smoothly frame to frame.
    """
    out = rgb.copy()
    h, w = out.shape[:2]
    s = max(w, h)                        # scale line/marker size by the longer side
    lw = max(2, round(s / 550))
    r_major, r_minor = max(3, round(s / 320)), max(2, round(s / 450))
    halo = (15, 15, 15)

    def vis(p, i):
        return i in shown and (not hide_oof or _in_frame(p, i))

    def px(p, i):
        return (int(p[i, 0] * w), int(p[i, 1] * h))

    region_rgb = {k: _hex_to_rgb(v) for k, v in REGION_COLORS.items()}
    # Pass 1 = dark halo underneath, pass 2 = the colored skeleton on top, so
    # the colors stay legible on any background (white sheets, skin, etc.).
    for colored in (False, True):
        for p in poses:
            for region, edges in CONNECTIONS.items():
                col = region_rgb[region] if colored else halo
                for a, b in edges:
                    if vis(p, a) and vis(p, b):
                        cv2.line(out, px(p, a), px(p, b), col,
                                 lw if colored else lw + 1, cv2.LINE_AA)
            for i in range(33):
                if vis(p, i):
                    r = r_major if i in MAJOR else r_minor
                    col = _hex_to_rgb(joint_color(i)) if colored else halo
                    cv2.circle(out, px(p, i), r if colored else r + 1, col,
                               -1, cv2.LINE_AA)
    return out


def build_figure(rgb, poses, shown, labels_mode="hover", hide_oof=True,
                 pinned=frozenset(), height=780):
    """rgb: (H,W,3) uint8. poses: list of (33,5). shown: set of landmark ids."""
    h, w = rgb.shape[:2]
    fig = go.Figure()

    fig.add_layout_image(dict(
        source=Image.fromarray(rgb), xref="x", yref="y",
        x=0, y=0, sizex=w, sizey=h, xanchor="left", yanchor="top",
        sizing="stretch", layer="below",
    ))

    def visible(p, i):
        return i in shown and (not hide_oof or _in_frame(p, i))

    for pi, p in enumerate(poses):
        # Skeleton edges, one trace per region (colored distinctly).
        for region, edges in CONNECTIONS.items():
            xs, ys = [], []
            for a, b in edges:
                if visible(p, a) and visible(p, b):
                    xs += [p[a, 0] * w, p[b, 0] * w, None]
                    ys += [p[a, 1] * h, p[b, 1] * h, None]
            if xs:
                # Dark halo underneath so the color stays visible on any background.
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color="rgba(0,0,0,0.55)", width=6),
                    hoverinfo="skip", showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=REGION_COLORS[region], width=3),
                    hoverinfo="skip", showlegend=(pi == 0),
                    name=region, legendgroup=region,
                ))

        # Joint markers (one trace) with hover = name + coords + visibility.
        jx, jy, jc, js, jt, jcustom = [], [], [], [], [], []
        for i in range(33):
            if not visible(p, i):
                continue
            jx.append(p[i, 0] * w)
            jy.append(p[i, 1] * h)
            jc.append(joint_color(i))
            js.append(10 if i in MAJOR else 7)
            jt.append("%s<br>x=%.3f y=%.3f<br>vis=%.2f"
                      % (DISPLAY_NAMES[i], p[i, 0], p[i, 1], p[i, 3]))
            jcustom.append(i)
        if jx:
            fig.add_trace(go.Scatter(
                x=jx, y=jy, mode="markers",
                marker=dict(color=jc, size=js,
                            line=dict(color="rgba(0,0,0,0.7)", width=1.5)),
                text=jt, hoverinfo="text", customdata=jcustom,
                showlegend=False, name="joints",
            ))

        # Labels: always-show for labelable joints, plus any pinned joint.
        lx, ly, lt = [], [], []
        for i in range(33):
            if not visible(p, i):
                continue
            if (labels_mode == "always" and i in LABELABLE) or i in pinned:
                lx.append(p[i, 0] * w)
                ly.append(p[i, 1] * h)
                lt.append(DISPLAY_NAMES[i])
        if lx:
            fig.add_trace(go.Scatter(
                x=lx, y=ly, mode="text", text=lt, textposition="top center",
                textfont=dict(size=11, color="#111"), hoverinfo="skip",
                showlegend=False,
            ))

        # Pinned joints: a highlight ring.
        px = [p[i, 0] * w for i in pinned if visible(p, i)]
        py = [p[i, 1] * h for i in pinned if visible(p, i)]
        if px:
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers",
                marker=dict(color="rgba(0,0,0,0)", size=16,
                            line=dict(color="#d97a00", width=2.5)),
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_xaxes(range=[0, w], visible=False)
    fig.update_yaxes(range=[h, 0], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=height,
    )
    return fig
