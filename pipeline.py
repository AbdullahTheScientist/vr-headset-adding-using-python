"""
VR Headset Pipeline — Automated batch processor.
Adds a realistic VR headset over the eye/upper-face region of every image
in the input folder while preserving the lower face (mouth, jaw, chin).

Usage
-----
  # Fast overlay mode (default, no API needed)
  python pipeline.py --input ./input --output ./output

  # AI inpainting mode (requires REPLICATE_API_TOKEN env var)
  python pipeline.py --input ./input --output ./output --mode ai

  # Single image
  python pipeline.py --input photo.jpg --output result.jpg

Options
-------
  --input   PATH   Input folder or single image file  [required]
  --output  PATH   Output folder or single image file [required]
  --mode    MODE   overlay (default) | ai
  --ext     EXTS   Comma-separated extensions to process (default: jpg,jpeg,png,webp)
  --workers INT    Parallel workers for overlay mode (default: 4)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from face_detector import FaceDetector
import overlay_engine
import real_compositor

# AI engine imported lazily to avoid hard dependency
_ai_inpaint = None


_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _load_ai():
    global _ai_inpaint
    if _ai_inpaint is None:
        from ai_engine import inpaint
        _ai_inpaint = inpaint
    return _ai_inpaint


def process_image(
    src: Path,
    dst: Path,
    mode: str = "overlay",
    detector: FaceDetector = None,
    pose_estimator=None,
) -> dict:
    """
    Process a single image. Returns a result dict with keys:
      path, status, message, elapsed_s
    """
    t0 = time.time()
    img = cv2.imread(str(src))
    if img is None:
        return dict(path=str(src), status="error", message="Could not read image", elapsed_s=0)

    region, mask = detector.detect(img, pose_estimator=pose_estimator)
    if region is None:
        return dict(path=str(src), status="skipped", message="No face detected", elapsed_s=time.time() - t0)

    try:
        if mode == "overlay":
            # photoreal real-photo compositor; fall back to the procedural
            # renderer only if the real asset is unavailable
            try:
                result = real_compositor.composite(img, region)
            except Exception:
                result = overlay_engine.composite(img, region)
        elif mode == "ai":
            inpaint = _load_ai()
            result = inpaint(img, region)
        else:
            return dict(path=str(src), status="error", message=f"Unknown mode: {mode}", elapsed_s=0)
    except Exception as exc:
        return dict(path=str(src), status="error", message=str(exc), elapsed_s=time.time() - t0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), result)
    if not ok:
        return dict(path=str(src), status="error", message=f"Could not write {dst}", elapsed_s=time.time() - t0)

    return dict(path=str(src), status="ok", message=str(dst), elapsed_s=time.time() - t0)


def _collect_images(src_path: Path, ext_set: set) -> list[Path]:
    if src_path.is_file():
        return [src_path]
    return sorted(p for p in src_path.rglob("*") if p.suffix.lower() in ext_set)


def run(args):
    src = Path(args.input)
    dst = Path(args.output)
    mode = args.mode
    ext_set = {f".{e.strip().lower()}" for e in args.ext.split(",")}
    workers = max(1, args.workers)

    if not src.exists():
        print(f"[error] Input path does not exist: {src}")
        sys.exit(1)

    images = _collect_images(src, ext_set)
    if not images:
        print(f"[error] No images found in {src} with extensions {ext_set}")
        sys.exit(1)

    single_file = src.is_file()

    # Head-pose estimator (6DRepNet360). Only needed for the overlay warp.
    pose_estimator = None
    pose_label = "n/a"
    if mode == "overlay":
        from head_pose import HeadPoseEstimator
        pose_estimator = HeadPoseEstimator()
        pose_label = "6DRepNet360" if pose_estimator.ready else "solvePnP (fallback)"
        # 6DRepNet360 runs on CPU torch; serialise to avoid thread oversubscription
        if pose_estimator.ready:
            workers = 1

    print(f"VR Headset Pipeline")
    print(f"  Mode    : {mode}")
    print(f"  Pose    : {pose_label}")
    print(f"  Images  : {len(images)}")
    print(f"  Input   : {src}")
    print(f"  Output  : {dst}")
    print()

    results = []

    def make_dst(p: Path) -> Path:
        if single_file:
            return dst
        rel = p.relative_to(src)
        return dst / rel

    if mode == "overlay" and workers > 1 and len(images) > 1:
        # Multi-threaded for overlay (MediaPipe is thread-safe in static mode)
        def _task(p):
            det = FaceDetector()
            r = process_image(p, make_dst(p), mode=mode, detector=det,
                              pose_estimator=pose_estimator)
            det.close()
            return r

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, p): p for p in images}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
                results.append(fut.result())
    else:
        # Sequential (AI mode, or when the CPU pose-net is active)
        with FaceDetector() as det:
            for p in tqdm(images, unit="img"):
                r = process_image(p, make_dst(p), mode=mode, detector=det,
                                  pose_estimator=pose_estimator)
                results.append(r)

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print()
    print(f"Done. {len(ok)} processed | {len(skipped)} skipped (no face) | {len(errors)} errors")

    if skipped:
        print("\nSkipped:")
        for r in skipped:
            print(f"  {r['path']} — {r['message']}")

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  {r['path']} — {r['message']}")

    if ok:
        avg_s = sum(r["elapsed_s"] for r in ok) / len(ok)
        print(f"\nAvg time per image: {avg_s:.2f}s")
        print(f"Output: {dst}")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Add VR headset to face photos (batch, local, scripted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, help="Input folder or image file")
    parser.add_argument("--output", "-o", required=True, help="Output folder or image file")
    parser.add_argument("--mode", "-m", default="overlay", choices=["overlay", "ai"],
                        help="Processing mode: overlay (fast, no API) | ai (photorealistic, needs REPLICATE_API_TOKEN)")
    parser.add_argument("--ext", default="jpg,jpeg,png,webp",
                        help="Comma-separated file extensions to process")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel workers for overlay mode (default: 4)")

    args = parser.parse_args()
    success = run(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
