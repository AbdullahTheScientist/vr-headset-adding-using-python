"""
Photoreal VR-headset compositor.

Instead of *drawing* a headset (which always reads as CGI), this composites a
real product photograph of an Oculus Rift CV1 (straight-on front view, alpha
cut-out) onto the face. Real materials, foam, fabric and studio highlights come
for free; the job here is to seat that photo into the target image:

  1. load + cache the real cut-out (visor + side arms; over-head strap optional)
  2. scale so the visor spans temple-to-temple
  3. relight: match the scene colour-temperature + exposure
  4. 3D-warp to head pose (reusing overlay_engine helpers)
  5. seat on skin: ambient occlusion + contact shadow + foam skin-bleed
  6. alpha-composite + gentle local grade / vignette

Only OpenCV, NumPy and Pillow — fully local, no network, no model.
"""
import functools
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

from overlay_engine import (
    analyze_lighting, _warp_3d_matrix, _warp_3d_euler,
    _place_alpha, _seat_on_skin, _grade_and_vignette, _apply_dark, _clamp,
)

_ASSET = Path(__file__).parent / "assets" / "real" / "rift_front2.png"


# visor-plate crop on the 750x500 source (measured): drops the splaying side
# arms, round pivots and over-head strap, leaving just the eye-covering face.
_VISOR_BOX = (0.149, 0.326, 0.851, 0.866)            # x1,y1,x2,y2 fractions


@functools.lru_cache(maxsize=3)
def _load_asset(variant: str):
    """Real headset cut-out as RGBA uint8, cropped per ``variant``."""
    arr = np.array(Image.open(_ASSET).convert("RGBA"))
    H, W = arr.shape[:2]

    if variant == "visor":
        fx1, fy1, fx2, fy2 = _VISOR_BOX
        arr = arr[int(fy1 * H):int(fy2 * H), int(fx1 * W):int(fx2 * W)]
        # rounded-rect mask hugging the face plate: trims the strap/pivot
        # fabric that lingers in the corners (esp. the bottom corners).
        ch, cw = arr.shape[:2]
        m = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(m).rounded_rectangle(
            [int(cw * 0.085), int(ch * 0.04), int(cw * 0.915), int(ch * 0.97)],
            radius=int(ch * 0.26), fill=255)
        m = m.filter(ImageFilter.GaussianBlur(max(1, cw // 220)))
        mm = np.array(m).astype(np.float32) / 255.0
        arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * mm).astype(np.uint8)
    else:  # "full" — drop only the over-head strap loop
        al = arr[:, :, 3]
        vis = np.where((al > 30).mean(axis=1) > 0.22)[0]
        if len(vis):
            arr = arr[max(0, vis[0] - 3):]

    al = arr[:, :, 3]
    cols = np.where((al > 20).any(axis=0))[0]
    rows = np.where((al > 20).any(axis=1))[0]
    arr = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return np.ascontiguousarray(arr)


def _relight(sprite_rgba, lighting, cast_strength=0.22):
    """Match the headset to the scene colour cast + overall exposure."""
    arr = sprite_rgba.astype(np.float32)
    rgb = arr[:, :, :3]                              # PIL = RGB
    al = arr[:, :, 3]
    m = al > 20

    cast = lighting["cast"]                          # BGR, mean 1.0
    cast_rgb = np.array([cast[2], cast[1], cast[0]], np.float32)
    rgb *= (1.0 + cast_strength * (cast_rgb - 1.0))[None, None, :]

    if m.any():
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        cur = max(float(lum[m].mean()), 1.0)
        target = _clamp(lighting["luma"] * 0.55, 26, 160)
        rgb *= _clamp(target / cur, 0.70, 1.30)

    arr[:, :, :3] = np.clip(rgb, 0, 255)
    return arr.astype(np.uint8)


def _paste_rgba(bg_bgr, sprite_pil, x, y):
    """Alpha-composite an RGBA PIL sprite onto a BGR array at (x, y), clipped."""
    sp = np.array(sprite_pil).astype(np.float32)
    sh, sw = sp.shape[:2]
    H, W = bg_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + sw), min(H, y + sh)
    if x1 <= x0 or y1 <= y0:
        return bg_bgr
    crop = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    a = crop[:, :, 3:4] / 255.0
    rgb_bgr = crop[:, :, 2::-1]                      # RGB -> BGR
    dst = bg_bgr[y0:y1, x0:x1].astype(np.float32)
    bg_bgr[y0:y1, x0:x1] = (dst * (1 - a) + rgb_bgr * a).astype(np.uint8)
    return bg_bgr


def composite(face_bgr, region, *, perspective=True,
              scale=1.06, y_bias=0.30, variant="visor"):
    """
    Render a photoreal VR headset over the face from a real product photo.
    face_bgr : H×W×3 uint8 BGR ;  region : dict from FaceDetector.detect()
    Returns BGR, same size.
    """
    asset = _load_asset(variant)
    ah, aw = asset.shape[:2]

    lighting = analyze_lighting(face_bgr, region)

    # --- scale so the headset spans (a bit past) temple-to-temple ----------
    target_w = max(8.0, region["width"] * scale)
    s = target_w / aw
    nw, nh = max(8, int(aw * s)), max(8, int(ah * s))
    sprite = Image.fromarray(asset, "RGBA").resize((nw, nh), Image.LANCZOS)
    sprite = Image.fromarray(_relight(np.array(sprite), lighting), "RGBA")

    # --- 3D warp to head pose ---------------------------------------------
    if perspective and region.get("R") is not None:
        warped, (ocx, ocy) = _warp_3d_matrix(sprite, region["R"])
    elif perspective:
        warped, (ocx, ocy) = _warp_3d_euler(
            sprite, region.get("yaw", 0.0), region.get("pitch", 0.0),
            region.get("angle", 0.0))
    else:
        warped, (ocx, ocy) = sprite, (nw / 2.0, nh / 2.0)

    # --- placement: visor centre over the eye line -------------------------
    cx = region["cx"]
    cy = int(region["cy"] + y_bias * region["height"])
    hs_x = int(cx - ocx)
    hs_y = int(cy - ocy)

    foot = _place_alpha(np.array(warped.split()[3]), (hs_x, hs_y), face_bgr.shape)

    # --- shadows / AO / skin bleed on the surrounding skin -----------------
    work_bgr = _seat_on_skin(face_bgr.copy(), foot, region, lighting)

    # --- crisp contact shadow hugging the foam's lower edge ----------------
    #     (grounds the device so its bottom doesn't look pasted-on)
    H, W = work_bgr.shape[:2]
    foot_u8 = (foot * 255).astype(np.uint8)
    drop = max(3, region["height"] // 16)
    M = np.float32([[1, 0, 0], [0, 1, drop]])
    shifted = cv2.warpAffine(foot_u8, M, (W, H)).astype(np.float32) / 255.0
    contact = np.clip(shifted - foot, 0, 1)               # only on skin below
    sig = max(1.0, region["width"] / 110.0)
    contact = cv2.GaussianBlur(contact, (0, 0), sig)
    contact = np.clip(contact * 0.55, 0, 0.55)
    work_bgr = _apply_dark(work_bgr, contact)

    # --- composite the headset + gentle local grade ------------------------
    work_bgr = _paste_rgba(work_bgr, warped, hs_x, hs_y)
    work_bgr = _grade_and_vignette(work_bgr, foot, region)
    return work_bgr
