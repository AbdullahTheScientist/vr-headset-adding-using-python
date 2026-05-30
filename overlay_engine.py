"""
Composites a VR headset onto a face image with photoreal compositing.

Pipeline per face:
  1. scale headset from interpupillary distance (IPD)
  2. sample scene lighting (warmth + direction) from forehead / cheeks
  3. tint headset to the scene colour cast
  4. paint a specular highlight on the lens toward the light source
  5. 3D-warp the headset to match head pose (yaw / pitch / roll)
  6. feather the side edges so it wraps the face
  7. lay side straps, contact shadow, ambient-occlusion and skin bleed
  8. alpha-composite with a soft (feathered) alpha
  9. gentle local colour-grade + vignette to seat it in the photo

Only OpenCV, NumPy and Pillow are used — fully local, no network.
"""
import math
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import cv2

from headset_renderer import get_headset


# ---------------------------------------------------------------------------
# 1. scene lighting analysis
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _sample_box(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, x2 = _clamp(x1, 0, w), _clamp(x2, 0, w)
    y1, y2 = _clamp(y1, 0, h), _clamp(y2, 0, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img[y1:y2, x1:x2]


def analyze_lighting(face_bgr, region):
    """
    Returns dict:
      skin_bgr  : median skin colour (3,) float
      cast      : per-channel colour cast (BGR) normalised to mean 1.0
      luma      : mean skin luminance 0-255
      light_dir : (dx, dy) unit-ish vector pointing toward the light,
                  in headset-local space (-1..1 each)
    """
    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    w_r = x2 - x1
    h_r = y2 - y1

    samples = []
    # forehead strip (above headset)
    fh = _sample_box(face_bgr,
                     x1 + int(w_r * 0.20), y1 - int(h_r * 0.55),
                     x2 - int(w_r * 0.20), y1 - int(h_r * 0.05))
    if fh is not None:
        samples.append(fh.reshape(-1, 3))
    # cheeks (below headset, flanking the nose)
    lc = _sample_box(face_bgr, x1, y2 + int(h_r * 0.05),
                     x1 + int(w_r * 0.22), y2 + int(h_r * 0.45))
    rc = _sample_box(face_bgr, x2 - int(w_r * 0.22), y2 + int(h_r * 0.05),
                     x2, y2 + int(h_r * 0.45))
    for c in (lc, rc):
        if c is not None:
            samples.append(c.reshape(-1, 3))

    if samples:
        allpx = np.concatenate(samples, axis=0).astype(np.float32)
        skin_bgr = np.median(allpx, axis=0)
    else:
        skin_bgr = np.array([150, 150, 150], dtype=np.float32)

    mean_ch = max(float(skin_bgr.mean()), 1.0)
    cast = skin_bgr / mean_ch                      # e.g. warm skin -> R>1, B<1
    luma = float(0.114 * skin_bgr[0] + 0.587 * skin_bgr[1] + 0.299 * skin_bgr[2])

    # light direction from forehead brightness centroid
    light_dir = (0.0, -0.6)                        # default: slightly from top
    if fh is not None:
        g = cv2.cvtColor(fh, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gh, gw = g.shape
        if gh > 2 and gw > 2:
            ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
            tot = g.sum() + 1e-6
            cxw = (g * xs).sum() / tot
            cyw = (g * ys).sum() / tot
            dx = (cxw - gw / 2) / (gw / 2)         # -1..1
            dy = (cyw - gh / 2) / (gh / 2)
            light_dir = (_clamp(dx, -1, 1), _clamp(dy - 0.4, -1, 1))

    return {"skin_bgr": skin_bgr, "cast": cast, "luma": luma, "light_dir": light_dir}


# ---------------------------------------------------------------------------
# 2. headset surface treatment (tint + specular) — on the flat sprite
# ---------------------------------------------------------------------------

def _tint_headset(headset_rgba, cast, luma, strength=0.30):
    """Nudge the neutral-grey headset toward the scene colour cast + exposure."""
    arr = np.array(headset_rgba).astype(np.float32)
    rgb = arr[:, :, :3]
    a = arr[:, :, 3:4]

    # colour cast: PIL is RGB, cast is BGR -> flip
    cast_rgb = np.array([cast[2], cast[1], cast[0]], dtype=np.float32)
    gain = 1.0 + strength * (cast_rgb - 1.0)
    rgb = rgb * gain[np.newaxis, np.newaxis, :]

    # exposure: a dark device in a bright scene stays dark but not crushed
    cur = max(rgb.mean(), 1.0)
    target = _clamp(luma * 0.42, 18, 150)
    exp = _clamp(target / cur, 0.6, 1.35)
    rgb = rgb * exp

    out = np.concatenate([np.clip(rgb, 0, 255), a], axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _add_specular(headset_rgba, light_dir, opacity=0.16):
    """
    Soft reflection streak on the glossy top surface, biased toward the light.
    Kept subtle and high on the body so it reads as a surface reflection, not
    a glow over the lenses.
    """
    w, h = headset_rgba.size
    spec = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(spec)

    dx, dy = light_dir
    # high on the body (glossy top), gently biased horizontally toward the light
    cx = int(w * (0.5 + 0.22 * dx))
    cy = int(h * 0.16)
    # wide, shallow streak — like a soft reflection of the environment
    ell_w = int(w * 0.26)
    ell_h = int(h * 0.05)
    d.ellipse([cx - ell_w, cy - ell_h, cx + ell_w, cy + ell_h], fill=255)

    blur_r = max(5, w // 70)
    spec = spec.filter(ImageFilter.GaussianBlur(blur_r))

    # clip to the headset body and modulate opacity
    hs_alpha = np.array(headset_rgba.split()[3]).astype(np.float32) / 255.0
    spec_arr = np.array(spec).astype(np.float32) / 255.0 * hs_alpha * opacity

    arr = np.array(headset_rgba).astype(np.float32)
    white = np.array([225, 230, 240], dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - spec_arr) + white[c] * spec_arr
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------
# 3. geometry: 3D warp + side feather
# ---------------------------------------------------------------------------

# 6DRepNet360 reports rotation in a camera frame with x-right, y-up,
# z-toward-camera. Our screen quad uses x-right, y-down, z-out-of-screen.
# This diagonal sign flip maps one into the other via Rs = S @ R @ S.
_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])


def _project_quad(headset_rgba, R_screen, f_factor=1.5):
    """
    Project the flat headset quad through a screen-space rotation matrix.
    Returns (warped_rgba, (ocx, ocy)) — the headset centre inside the warp.
    """
    img = np.array(headset_rgba)
    h, w = img.shape[:2]

    corners = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0],
                        [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=np.float64)
    f = w * f_factor
    d = f
    proj = []
    for c in corners:
        X, Y, Z = R_screen @ c
        s = f / (d - Z)
        proj.append([X * s, Y * s])
    proj = np.array(proj, dtype=np.float32)

    minx, miny = proj.min(axis=0)
    maxx, maxy = proj.max(axis=0)
    proj_shift = proj - [minx, miny]
    out_w = int(math.ceil(maxx - minx))
    out_h = int(math.ceil(maxy - miny))
    ocx, ocy = float(-minx), float(-miny)          # centre lands here

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, proj_shift)
    warped = cv2.warpPerspective(img, M, (max(out_w, 1), max(out_h, 1)),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    return Image.fromarray(warped, "RGBA"), (ocx, ocy)


def _damp_rotation(R, damp=0.85, max_deg=42.0):
    """Scale a rotation toward identity (axis-angle) and cap its magnitude."""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-6:
        return np.eye(3)
    axis = rvec / theta
    theta = _clamp(theta * damp, -math.radians(max_deg), math.radians(max_deg))
    R2, _ = cv2.Rodrigues(axis * theta)
    return R2


def _warp_3d_matrix(headset_rgba, R):
    """Warp using a 3x3 rotation matrix (6DRepNet360) mapped straight in."""
    Rd = _damp_rotation(np.asarray(R, dtype=np.float64))
    R_screen = _AXIS_FLIP @ Rd @ _AXIS_FLIP
    return _project_quad(headset_rgba, R_screen)


def _warp_3d_euler(headset_rgba, yaw, pitch, roll, damp=0.8, max_ang=28.0):
    """Fallback warp from Euler angles (solvePnP path)."""
    yaw = math.radians(_clamp(yaw * damp, -max_ang, max_ang))
    pitch = math.radians(_clamp(pitch * damp, -max_ang, max_ang))
    rollr = math.radians(_clamp(roll, -35, 35))

    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cx_, sx_ = math.cos(pitch), math.sin(pitch)
    cz_, sz_ = math.cos(rollr), math.sin(rollr)

    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
    Rz = np.array([[cz_, -sz_, 0], [sz_, cz_, 0], [0, 0, 1]])
    return _project_quad(headset_rgba, Rz @ Ry @ Rx)


def _feather_sides(img, fade_pct=0.07):
    w, h = img.size
    arr = np.array(img).copy()
    alpha = arr[:, :, 3].astype(np.float32)
    fade_w = int(w * fade_pct)
    if fade_w >= 2:
        ramp = np.linspace(0, 1, fade_w, dtype=np.float32)
        alpha[:, :fade_w] *= ramp[np.newaxis, :]
        alpha[:, w - fade_w:] *= ramp[::-1][np.newaxis, :]
        arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------------------
# 4. side straps
# ---------------------------------------------------------------------------

def _draw_side_arms(canvas, hs_x, hs_y, sprite, region, img_w, img_h):
    """
    Slim side straps that taper and curve gently downward toward the ears,
    the way a real headset band wraps the head. Each strap is stamped as a
    chain of discs along a parabolic centre-line (horizontal where it meets
    the face plate, curving down at the far end) so it ends in a rounded
    cap instead of a flat, belt-like edge.
    """
    sp_h = sprite.height
    arm_h = max(6, int(sp_h * 0.15))            # slimmer band (was 0.26)
    arm_yc = hs_y + int(sp_h * 0.44)

    face_w = region["x2"] - region["x1"]
    reach = max(22, int(face_w * 0.22))         # shorter reach (was 0.28)
    overlap = max(6, int(face_w * 0.04))
    curve_drop = max(4, int(arm_h * 1.3))       # how far the end curves down

    top_c = np.array([62, 63, 70], np.float32)
    bot_c = np.array([26, 26, 30], np.float32)
    spec_c = np.array([95, 97, 105], np.float32)
    edge_c = np.array([16, 16, 20], np.float32)

    for side in ("left", "right"):
        if side == "left":
            x_inner = _clamp(region["x1"] + overlap, 0, img_w)
            x_outer = _clamp(region["x1"] - reach, 0, img_w)
        else:
            x_inner = _clamp(region["x2"] - overlap, 0, img_w)
            x_outer = _clamp(region["x2"] + reach, 0, img_w)
        if abs(x_outer - x_inner) <= 4:
            continue

        # local canvas covering the curved band, clamped on screen
        pad = arm_h
        bx_min = max(0, min(x_inner, x_outer) - pad)
        bx_max = min(img_w, max(x_inner, x_outer) + pad)
        by_min = max(0, arm_yc - arm_h - pad)
        by_max = min(img_h, arm_yc + curve_drop + arm_h + pad)
        bw, bh = bx_max - bx_min, by_max - by_min
        if bw <= 2 or bh <= 2:
            continue

        # stamp discs along a parabolic, tapering centre-line
        mask = Image.new("L", (bw, bh), 0)
        md = ImageDraw.Draw(mask)
        steps = max(24, int(abs(x_outer - x_inner)))
        for i in range(steps + 1):
            t = i / steps
            cx = x_inner + (x_outer - x_inner) * t - bx_min
            cy = arm_yc + curve_drop * (t * t) - by_min        # flat → down
            rad = (arm_h / 2.0) * (1.0 - 0.5 * t ** 1.3)       # taper to ~50%
            md.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(1, arm_h // 10)))
        mask_arr = np.array(mask).astype(np.float32)
        if mask_arr.max() < 1:
            continue

        # cylindrical shading that follows the curve: shade by each column's
        # vertical position within the band (top bright, bottom dark)
        present = mask_arr > 20
        ys = np.arange(bh, dtype=np.float32)[:, None]
        ymin = np.where(present, ys, np.inf).min(axis=0)
        ymax = np.where(present, ys, -np.inf).max(axis=0)
        denom = np.clip(ymax - ymin, 1, None)
        tv = np.clip((ys - ymin[None, :]) / denom[None, :], 0, 1)

        grad = (1 - tv)[..., None] * top_c + tv[..., None] * bot_c
        spec_band = np.clip(1 - tv * 6.0, 0, 1)[..., None]      # top sliver
        edge_band = np.clip((tv - 0.82) / 0.18, 0, 1)[..., None]
        grad = grad * (1 - spec_band) + spec_c * spec_band
        grad = grad * (1 - edge_band) + edge_c * edge_band

        arr = np.zeros((bh, bw, 4), np.float32)
        arr[:, :, :3] = grad
        arr[:, :, 3] = mask_arr
        canvas.alpha_composite(Image.fromarray(arr.astype(np.uint8), "RGBA"),
                               dest=(bx_min, by_min))
    return canvas


# ---------------------------------------------------------------------------
# 5. shadows, ambient occlusion, skin bleed (operate on full-image footprint)
# ---------------------------------------------------------------------------

def _place_alpha(sprite_alpha, top_left, img_shape):
    """Paste a sprite's alpha onto a full-image canvas. Returns float 0-1 (H,W)."""
    H, W = img_shape[:2]
    out = np.zeros((H, W), np.float32)
    sh, sw = sprite_alpha.shape
    x, y = top_left
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + sw), min(H, y + sh)
    if x1 <= x0 or y1 <= y0:
        return out
    sx0, sy0 = x0 - x, y0 - y
    out[y0:y1, x0:x1] = sprite_alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)] / 255.0
    return out


def _apply_dark(canvas_bgr, dark_alpha, color=(0, 0, 0)):
    """Multiply a dark layer (alpha 0-1) onto a BGR image in place-ish."""
    col = np.array(color, np.float32)
    a = dark_alpha[..., None]
    return (canvas_bgr.astype(np.float32) * (1 - a) + col * a).astype(np.uint8)


def _seat_on_skin(canvas_bgr, foot, region, lighting):
    """
    foot : full-image headset alpha 0-1
    Adds (a) ambient-occlusion gradient under the bottom edge,
         (b) thin contact shadow around the perimeter,
         (c) subtle skin-tone bleed just under the edge.
    Returns updated BGR.
    """
    H, W = canvas_bgr.shape[:2]
    foot_u8 = (foot * 255).astype(np.uint8)
    k = max(3, (region["width"] // 12) | 1)

    # (a) ambient occlusion below the headset: shift footprint down, blur, keep
    #     only the part that falls outside the headset (on skin)
    shift = max(6, region["height"] // 8)
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    shifted = cv2.warpAffine(foot_u8, M, (W, H))
    ao = cv2.GaussianBlur(shifted, (k | 1, k | 1), region["width"] / 60.0)
    ao = ao.astype(np.float32) / 255.0
    ao = ao * (1.0 - foot)                         # only on skin, not on device
    ao = np.clip(ao * 0.42, 0, 0.42)               # 30-50% range, capped
    canvas_bgr = _apply_dark(canvas_bgr, ao)

    # (b) thin contact shadow: dilate - original, ~3px, darker
    ksz = max(3, region["width"] // 200)
    kernel = np.ones((ksz * 2 + 1, ksz * 2 + 1), np.uint8)
    dil = cv2.dilate(foot_u8, kernel).astype(np.float32) / 255.0
    ring = np.clip(dil - foot, 0, 1)
    ring = cv2.GaussianBlur(ring, (5, 5), 1.5) * 0.5
    canvas_bgr = _apply_dark(canvas_bgr, ring)

    # (c) skin bleed: soft skin-tone band just outside the lower edge,
    #     simulating the foam gasket pressing into the skin
    skin = lighting["skin_bgr"]
    band = np.clip(dil - foot, 0, 1)
    band = cv2.GaussianBlur(band, (k | 1, k | 1), region["width"] / 80.0)
    # keep the band mostly on the lower half of the device
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    lower_bias = np.clip((yy - region["cy"] / H) * 3 + 0.3, 0, 1)
    band = band * lower_bias * 0.25
    a = band[..., None]
    canvas_bgr = (canvas_bgr.astype(np.float32) * (1 - a) +
                  skin[None, None, :] * a).astype(np.uint8)
    return canvas_bgr


# ---------------------------------------------------------------------------
# 6. local colour grade + vignette over the headset
# ---------------------------------------------------------------------------

def _grade_and_vignette(canvas_bgr, foot, region):
    """Subtle CLAHE local-contrast match + radial vignette, masked to the device."""
    H, W = canvas_bgr.shape[:2]
    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    pad = region["width"] // 6
    bx1, by1 = _clamp(x1 - pad, 0, W), _clamp(y1 - pad, 0, H)
    bx2, by2 = _clamp(x2 + pad, 0, W), _clamp(y2 + pad, 0, H)
    if bx2 - bx1 < 4 or by2 - by1 < 4:
        return canvas_bgr

    crop = canvas_bgr[by1:by2, bx1:bx2].astype(np.uint8)
    foot_crop = foot[by1:by2, bx1:bx2]

    # CLAHE on L channel for gentle local contrast/grain match
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    graded = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # radial vignette centred on the crop
    ch, cw = crop.shape[:2]
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    r = np.sqrt(((xx - cw / 2) / (cw / 2)) ** 2 + ((yy - ch / 2) / (ch / 2)) ** 2)
    vig = np.clip(1.0 - 0.18 * np.clip(r - 0.4, 0, 1), 0.82, 1.0)[..., None]
    graded = graded * vig

    # blend graded result back only where the device is (soft alpha)
    a = (foot_crop * 0.6)[..., None]               # 60% strength, keeps subtle
    out = crop.astype(np.float32) * (1 - a) + graded * a
    canvas_bgr[by1:by2, bx1:bx2] = np.clip(out, 0, 255).astype(np.uint8)
    return canvas_bgr


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def composite(face_bgr, region, *, perspective=True):
    """
    Renders and composites a photoreal VR headset over the face.
    face_bgr : H×W×3 uint8 BGR ;  region : dict from FaceDetector.detect()
    Returns BGR same size as face_bgr.
    """
    h, w = face_bgr.shape[:2]

    # --- size from IPD (physically plausible), bounded by the bbox ---------
    ipd = region.get("ipd", 0) or (region["width"] / 2.5)
    hs_w = int(_clamp(ipd * 2.55, region["width"] * 0.85, region["width"] * 1.25))
    hs_h = int(hs_w * 0.42)

    # --- lighting ----------------------------------------------------------
    lighting = analyze_lighting(face_bgr, region)

    # --- build + treat the flat sprite -------------------------------------
    headset = get_headset(hs_w, hs_h)
    headset = _tint_headset(headset, lighting["cast"], lighting["luma"])
    headset = _add_specular(headset, lighting["light_dir"])

    # --- 3D warp to head pose ---------------------------------------------
    if perspective:
        R = region.get("R", None)
        if R is not None:
            # 6DRepNet360: map the rotation matrix straight into the warp
            sprite, (ocx, ocy) = _warp_3d_matrix(headset, R)
        else:
            # solvePnP fallback via Euler angles
            sprite, (ocx, ocy) = _warp_3d_euler(
                headset, region.get("yaw", 0.0), region.get("pitch", 0.0),
                region.get("angle", 0.0))
    else:
        sprite, (ocx, ocy) = headset, (headset.width / 2, headset.height / 2)

    sprite = _feather_sides(sprite, fade_pct=0.06)

    # placement so the sprite centre lands on the region centre
    hs_x = int(region["cx"] - ocx)
    hs_y = int(region["cy"] - ocy)

    # full-image footprint of the (warped) headset alpha
    foot = _place_alpha(np.array(sprite.split()[3]), (hs_x, hs_y), face_bgr.shape)

    # --- assemble on a PIL canvas -----------------------------------------
    canvas = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # straps go first so the face plate covers their inner ends
    canvas = _draw_side_arms(canvas, hs_x, hs_y, sprite, region, w, h)

    # bake straps into BGR so the shadow/grade passes see them
    work_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)

    # shadows / AO / skin bleed on the skin around the device
    work_bgr = _seat_on_skin(work_bgr, foot, region, lighting)

    # composite the headset itself (soft alpha)
    canvas = Image.fromarray(cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    canvas.alpha_composite(sprite, dest=(hs_x, hs_y))
    work_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)

    # gentle grade + vignette to seat it in the photo
    work_bgr = _grade_and_vignette(work_bgr, foot, region)

    return work_bgr
