"""Quick iteration harness for the real-photo compositor."""
import sys
from pathlib import Path
import cv2

from face_detector import FaceDetector
import real_compositor as rc

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.06
y_bias = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
variant = sys.argv[3] if len(sys.argv) > 3 else "visor"
outdir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("output_real")

try:
    from head_pose import HeadPoseEstimator
    pe = HeadPoseEstimator()
except Exception as e:
    print("pose est unavailable:", e)
    pe = None

outdir.mkdir(exist_ok=True)
det = FaceDetector()
for p in sorted(Path("input").glob("*.jpg")):
    img = cv2.imread(str(p))
    region, _ = det.detect(img, pose_estimator=pe)
    if region is None:
        print("no face:", p.name)
        continue
    res = rc.composite(img, region, scale=scale, y_bias=y_bias,
                       variant=variant)
    cv2.imwrite(str(outdir / p.name), res)
    print(f"ok {p.name}  yaw~{region.get('yaw',0):.0f} pose={region['pose_src']}")
det.close()
print(f"-> {outdir}  (scale={scale} y_bias={y_bias} variant={variant})")
