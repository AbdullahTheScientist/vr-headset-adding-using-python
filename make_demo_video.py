"""
Creates a demo MP4 showing before/after wipe-reveal for each face image.
Output: demo.mp4
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── config ────────────────────────────────────────────────────────────────
FPS          = 30
W, H         = 1024, 1024
OUT_PATH     = "demo.mp4"
INPUT_DIR    = Path("input")
OUTPUT_DIR   = Path("output")

FONT_REG  = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"

# Frames per phase (per image)
F_HOLD_BEFORE = 35    # show "BEFORE" still
F_WIPE        = 80    # wipe line travels left → right
F_HOLD_AFTER  = 45    # show "AFTER" still
F_FADE        = 18    # fade-to-black between images

# Title card
F_TITLE       = 55

# ── helpers ───────────────────────────────────────────────────────────────

def ease_inout(t: float) -> float:
    """Smooth-step easing."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def load_pair(name: str):
    before = cv2.imread(str(INPUT_DIR / name))
    after  = cv2.imread(str(OUTPUT_DIR / name))
    if before is None or after is None:
        return None, None
    before = cv2.resize(before, (W, H))
    after  = cv2.resize(after,  (W, H))
    return before, after


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def draw_label(pil_img: Image.Image, text: str, x: int, y: int,
               font_path: str, size: int,
               color=(255, 255, 255), shadow=True) -> Image.Image:
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, size)
    if shadow:
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=color)
    return pil_img


def text_size(text: str, font_path: str, size: int):
    font = ImageFont.truetype(font_path, size)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bb = dummy.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def build_wipe_frame(before: np.ndarray, after: np.ndarray,
                     t: float, label_alpha: float = 1.0) -> np.ndarray:
    """
    t=0 → all before;  t=1 → all after.
    Draws wipe line + BEFORE/AFTER labels that fade with the reveal.
    """
    x = int(ease_inout(t) * W)
    frame = before.copy()
    if x > 0:
        frame[:, :x] = after[:, :x]

    # Glowing wipe line
    if 0 < x < W:
        cv2.line(frame, (x, 0), (x, H), (255, 255, 255), 4)
        for glow_w, alpha in [(10, 60), (20, 30)]:
            overlay = frame.copy()
            cv2.line(overlay, (x, 0), (x, H), (220, 220, 255), glow_w)
            cv2.addWeighted(overlay, alpha / 255, frame, 1 - alpha / 255, 0, frame)

    # Labels
    pil = bgr_to_pil(frame)
    a = int(255 * label_alpha)

    # "BEFORE" — left half
    before_x = max(12, x // 2 - 60)
    if x < W - 20:
        draw_label(pil, "BEFORE", before_x, 18,
                   FONT_BOLD, 52, color=(255, 255, 255, a))

    # "AFTER" — right half
    after_label_x = x + max(12, (W - x) // 2 - 55)
    if x > 20:
        draw_label(pil, "AFTER", after_label_x, 18,
                   FONT_BOLD, 52, color=(100, 220, 120, a))

    return pil_to_bgr(pil)


def make_title_card(frame_idx: int, total: int) -> np.ndarray:
    """Fades in/out a dark title card."""
    t_in  = min(1.0, frame_idx / 12)
    t_out = min(1.0, (total - frame_idx) / 12)
    alpha = ease_inout(min(t_in, t_out))

    img = Image.new("RGB", (W, H), (12, 14, 18))
    draw = ImageDraw.Draw(img)

    # Subtle gradient background
    for y in range(H):
        v = int(12 + 8 * y / H)
        draw.line([(0, y), (W, y)], fill=(v, v + 2, v + 4))

    # Title
    title   = "VR Headset Pipeline"
    t_w, _  = text_size(title, FONT_BOLD, 64)
    draw_label(img, title, (W - t_w) // 2, H // 2 - 70,
               FONT_BOLD, 64, color=(240, 240, 245))

    # Subtitle
    sub    = "Photoreal • Local • No API"
    s_w, _ = text_size(sub, FONT_REG, 32)
    draw_label(img, sub, (W - s_w) // 2, H // 2 + 10,
               FONT_REG, 32, color=(140, 200, 160))

    # Separator line
    lx = (W - 380) // 2
    draw.rectangle([lx, H // 2 - 10, lx + 380, H // 2 - 7], fill=(80, 80, 90))

    arr = pil_to_bgr(img)

    # Fade in/out
    black = np.zeros_like(arr)
    return cv2.addWeighted(arr, alpha, black, 1 - alpha, 0)


def make_transition(before_after_frame: np.ndarray, fade_idx: int, total: int) -> np.ndarray:
    """Fade to black."""
    alpha = 1.0 - ease_inout(fade_idx / max(total - 1, 1))
    return (before_after_frame.astype(np.float32) * alpha).astype(np.uint8)


def watermark(frame: np.ndarray) -> np.ndarray:
    pil = bgr_to_pil(frame)
    wm  = "VR Headset Pipeline — Before / After"
    ww, wh = text_size(wm, FONT_REG, 22)
    draw_label(pil, wm, W - ww - 10, H - wh - 10,
               FONT_REG, 22, color=(200, 200, 200), shadow=True)
    return pil_to_bgr(pil)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    images = sorted(p.name for p in INPUT_DIR.glob("*.jpg")
                    if (OUTPUT_DIR / p.name).exists())
    if not images:
        print("No matching input/output pairs found.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

    total_frames = (F_TITLE +
                    len(images) * (F_HOLD_BEFORE + F_WIPE + F_HOLD_AFTER + F_FADE))
    print(f"Rendering {total_frames} frames @ {FPS} fps -> {OUT_PATH}")

    # Title card
    for fi in range(F_TITLE):
        writer.write(watermark(make_title_card(fi, F_TITLE)))

    for idx, name in enumerate(images):
        before, after = load_pair(name)
        if before is None:
            continue

        stem = Path(name).stem.replace("_", " ").title()
        print(f"  [{idx+1}/{len(images)}] {stem}")

        # Hold BEFORE
        before_frame = build_wipe_frame(before, after, 0.0)
        for _ in range(F_HOLD_BEFORE):
            writer.write(watermark(before_frame))

        # Wipe reveal
        for fi in range(F_WIPE):
            t = fi / max(F_WIPE - 1, 1)
            writer.write(watermark(build_wipe_frame(before, after, t)))

        # Hold AFTER
        after_frame = build_wipe_frame(before, after, 1.0)
        for _ in range(F_HOLD_AFTER):
            writer.write(watermark(after_frame))

        # Fade to black
        for fi in range(F_FADE):
            writer.write(make_transition(after_frame, fi, F_FADE))

    writer.release()
    size_mb = Path(OUT_PATH).stat().st_size / 1024 / 1024
    duration = total_frames / FPS
    print(f"\nDone. {duration:.1f}s | {size_mb:.1f} MB -> {OUT_PATH}")


if __name__ == "__main__":
    main()
