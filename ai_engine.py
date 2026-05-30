"""
AI-powered VR headset inpainting via Replicate API.
Uses Stable Diffusion Inpainting to generate photorealistic results.
Falls back to overlay mode on errors.
"""
import os
import io
import base64
import tempfile
import requests
import numpy as np
import cv2
from PIL import Image

try:
    import replicate
    _HAS_REPLICATE = True
except ImportError:
    _HAS_REPLICATE = False


_INPAINT_MODEL = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"

_PROMPT = (
    "a person wearing a sleek modern Meta Quest 3 VR headset, "
    "matte dark gray, photorealistic, ultra detailed, high quality, "
    "natural lighting, 8k"
)
_NEGATIVE_PROMPT = (
    "blurry, cartoon, anime, low quality, distorted, unrealistic, "
    "helmet, goggles, sunglasses, swimming goggles"
)


def _img_to_b64_url(img_pil: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{data}"


def _build_inpaint_mask(region: dict, shape: tuple) -> Image.Image:
    """White = inpaint area, Black = keep."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    # Expand mask slightly for a clean fill
    pad = int((x2 - x1) * 0.05)
    x1m = max(0, x1 - pad)
    x2m = min(w, x2 + pad)
    y1m = max(0, y1 - pad)
    y2m = min(h, y2 + pad)
    cv2.rectangle(mask, (x1m, y1m), (x2m, y2m), 255, -1)
    blur_k = max(3, (x2m - x1m) // 8) | 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    return Image.fromarray(mask)


def inpaint(face_bgr: np.ndarray, region: dict) -> np.ndarray:
    """
    Returns BGR image with AI-generated VR headset.
    Raises RuntimeError if Replicate key not set or call fails.
    """
    if not _HAS_REPLICATE:
        raise RuntimeError("replicate package not installed. Run: pip install replicate")

    api_token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not api_token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN not set. "
            "Get a free key at https://replicate.com and set:\n"
            "  $env:REPLICATE_API_TOKEN = 'your_key_here'"
        )

    h, w = face_bgr.shape[:2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    # Resize to 512×512 (SD inpainting standard) keeping ratio
    target = 512
    ratio = target / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    # Pad to 512×512
    face_resized = face_pil.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (target, target), (0, 0, 0))
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    padded.paste(face_resized, (pad_x, pad_y))

    # Build mask (scaled + padded)
    scaled_region = {
        "x1": int(region["x1"] * ratio) + pad_x,
        "y1": int(region["y1"] * ratio) + pad_y,
        "x2": int(region["x2"] * ratio) + pad_x,
        "y2": int(region["y2"] * ratio) + pad_y,
    }
    mask_pil = _build_inpaint_mask(scaled_region, (target, target))

    client = replicate.Client(api_token=api_token)
    output = client.run(
        _INPAINT_MODEL,
        input={
            "prompt": _PROMPT,
            "negative_prompt": _NEGATIVE_PROMPT,
            "image": _img_to_b64_url(padded),
            "mask": _img_to_b64_url(mask_pil),
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "strength": 0.95,
        },
    )

    # output is a list of URLs
    if not output:
        raise RuntimeError("Replicate returned empty output")

    result_url = output[0] if isinstance(output, list) else str(output)
    resp = requests.get(result_url, timeout=60)
    resp.raise_for_status()

    result_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")

    # Crop padding and restore original size
    result_cropped = result_pil.crop((pad_x, pad_y, pad_x + new_w, pad_y + new_h))
    result_full = result_cropped.resize((w, h), Image.LANCZOS)

    # Composite: only replace the headset region, keep lower face from original
    result_arr = np.array(result_full)
    original_arr = face_rgb.copy()

    # Build blend mask: full alpha in headset region, original elsewhere
    blend_mask = np.zeros((h, w), dtype=np.float32)
    r = scaled_region  # NOTE: these are scaled coords, recompute from original
    rx1, rx2 = region["x1"], region["x2"]
    ry1, ry2 = region["y1"], region["y2"]
    blend_mask[ry1:ry2, rx1:rx2] = 1.0
    blur_k = max(3, (rx2 - rx1) // 6) | 1
    blend_mask = cv2.GaussianBlur(blend_mask, (blur_k, blur_k), 0)
    blend_mask = blend_mask[..., np.newaxis]  # (H,W,1)

    blended = (result_arr * blend_mask + original_arr * (1 - blend_mask)).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
