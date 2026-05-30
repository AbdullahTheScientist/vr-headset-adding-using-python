"""
Procedural VR headset renderer — Meta Quest 3 style, enhanced realism.
"""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def _body_mask(width: int, height: int, radius: int) -> np.ndarray:
    """Returns uint8 (H,W) mask of the rounded-rect body."""
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).rounded_rectangle([0, 0, width - 1, height - 1],
                                          radius=radius, fill=255)
    return np.array(img)


def _radial_gradient_rgba(w, h, inner, outer):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    t = np.clip(dist / (max(cx, cy) * 1.1), 0, 1)[..., np.newaxis]
    return ((1 - t) * np.array(inner) + t * np.array(outer)).astype(np.float32)


def _vertical_gradient(w, h, top, bot):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis, np.newaxis]
    return ((1 - t) * np.array(top) + t * np.array(bot)).astype(np.float32)


def _soft_noise(w, h, blur_r, amplitude):
    """Low-frequency grayscale noise for surface microstructure."""
    raw = np.random.default_rng(42).standard_normal((h, w)).astype(np.float32)
    noise_img = Image.fromarray(np.clip(raw * 40 + 128, 0, 255).astype(np.uint8), "L")
    noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
    noise = (np.array(noise_img).astype(np.float32) - 128) / 128.0  # [-1, 1]
    return noise * amplitude  # scalar offset for each pixel


def _draw_lens(draw, cx, cy, r, *, glass_color=(10, 12, 18), ring_color=(28, 28, 34)):
    """Draws a single outward-facing camera lens."""
    # Outer metal ring
    draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill=ring_color)
    # Inner dark glass
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=glass_color)
    # Specular highlight (top-left arc)
    hl_r = max(1, r // 3)
    hl_x = cx - r // 3
    hl_y = cy - r // 3
    hl_color = (60, 90, 140, 180)
    draw.ellipse([hl_x, hl_y, hl_x + hl_r, hl_y + hl_r], fill=hl_color[:3])
    # Tiny bright reflection dot
    dot_r = max(1, r // 5)
    draw.ellipse([cx - r // 2, cy - r // 2,
                  cx - r // 2 + dot_r, cy - r // 2 + dot_r],
                 fill=(200, 220, 255))


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_headset(width: int, height: int) -> Image.Image:
    w, h = width, height
    body_radius = int(h * 0.24)

    # -- Step 1: base gradient (vertical, light top → dark bottom) ----------
    base = _vertical_gradient(w, h,
                               top=(58, 60, 66, 255),
                               bot=(24, 24, 28, 255))

    # -- Step 2: radial highlight (convex surface illusion) ------------------
    radial = _radial_gradient_rgba(w, h,
                                    inner=(80, 82, 90, 50),
                                    outer=(0, 0, 0, 0))
    base_rgb = base[:, :, :3] + radial[:, :, :3]

    # -- Step 3: surface microstructure noise --------------------------------
    noise = _soft_noise(w, h, blur_r=max(2, w // 60), amplitude=6)
    base_rgb += noise[..., np.newaxis]

    # -- Step 4: assemble RGBA with body mask --------------------------------
    mask = _body_mask(w, h, body_radius)
    canvas_arr = np.zeros((h, w, 4), dtype=np.float32)
    canvas_arr[:, :, :3] = base_rgb
    canvas_arr[:, :, 3] = mask.astype(np.float32)
    canvas = _np_to_pil(canvas_arr)
    draw = ImageDraw.Draw(canvas)

    # -- Step 5: top-edge gloss strip ----------------------------------------
    gs_h = max(2, int(h * 0.018))
    gs_inset = int(w * 0.14)
    draw.rounded_rectangle(
        [gs_inset, int(h * 0.045), w - gs_inset, int(h * 0.045) + gs_h],
        radius=gs_h, fill=(160, 165, 175, 110),
    )

    # -- Step 6: subtle seam lines (panel divisions) -------------------------
    seam_y = int(h * 0.62)
    seam_color = (18, 18, 22, 200)
    draw.line([(int(w * 0.12), seam_y), (int(w * 0.88), seam_y)],
              fill=seam_color, width=max(1, h // 55))

    # -- Step 7: foam padding strip (bottom) ---------------------------------
    foam_top = int(h * 0.72)
    foam_radius = int(h * 0.09)
    # Foam body
    draw.rounded_rectangle(
        [int(w * 0.03), foam_top, int(w * 0.97), h - 1],
        radius=foam_radius, fill=(44, 42, 40, 255),
    )
    # Foam top highlight (stitching line look)
    draw.rounded_rectangle(
        [int(w * 0.08), foam_top + 3,
         int(w * 0.92), foam_top + max(2, int(h * 0.022))],
        radius=2, fill=(70, 66, 62, 150),
    )
    # Foam bottom edge shadow
    draw.rounded_rectangle(
        [int(w * 0.05), h - max(3, int(h * 0.025)),
         int(w * 0.95), h - 1],
        radius=foam_radius // 2, fill=(20, 19, 18, 220),
    )

    # -- Step 8: camera / sensor cluster (Quest 3 style — 3 lenses) ---------
    cam_y = int(h * 0.36)
    cam_r = max(5, int(w * 0.028))
    for cam_x in [int(w * 0.26), int(w * 0.50), int(w * 0.74)]:
        _draw_lens(draw, cam_x, cam_y, cam_r,
                   glass_color=(8, 10, 15),
                   ring_color=(30, 30, 36))

    # -- Step 9: IR emitter dots (small, flanking cameras) ------------------
    ir_r = max(2, int(w * 0.012))
    ir_y = cam_y
    for ir_x in [int(w * 0.12), int(w * 0.38), int(w * 0.62), int(w * 0.88)]:
        draw.ellipse([ir_x - ir_r, ir_y - ir_r, ir_x + ir_r, ir_y + ir_r],
                     fill=(20, 20, 26, 200))

    # -- Step 10: side strap bracket nubs -----------------------------------
    nub_w = max(5, int(w * 0.032))
    nub_h = int(h * 0.38)
    nub_y = int(h * 0.28)
    for nub_x in [0, w - nub_w]:
        # Main nub
        draw.rounded_rectangle(
            [nub_x, nub_y, nub_x + nub_w, nub_y + nub_h],
            radius=nub_w // 2, fill=(48, 48, 54, 255),
        )
        # Screw/rivet dot
        sc_r = max(1, nub_w // 4)
        sc_cx = nub_x + nub_w // 2
        sc_cy = nub_y + nub_h // 2
        draw.ellipse([sc_cx - sc_r, sc_cy - sc_r,
                      sc_cx + sc_r, sc_cy + sc_r],
                     fill=(35, 35, 40, 255))

    # -- Step 11: logo area (subtle embossed rectangle in centre) -----------
    logo_w, logo_h = int(w * 0.12), int(h * 0.06)
    logo_x = w // 2 - logo_w // 2
    logo_y = int(h * 0.20)
    draw.rounded_rectangle(
        [logo_x, logo_y, logo_x + logo_w, logo_y + logo_h],
        radius=logo_h // 3, fill=(42, 43, 48, 180),
    )

    # -- Step 12: re-apply body mask to clip all drawing --------------------
    canvas.putalpha(Image.fromarray(mask))

    # Feather edges very slightly
    alpha = np.array(canvas.split()[3]).astype(np.float32)
    alpha = np.array(
        Image.fromarray(alpha.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=max(1, w // 180))
        )
    ).astype(np.float32)
    canvas.putalpha(Image.fromarray(alpha.astype(np.uint8)))

    return canvas


def get_headset(width: int, height: int) -> Image.Image:
    return render_headset(max(width, 10), max(height, 6))
