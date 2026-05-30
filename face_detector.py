"""
Face landmark detection using MediaPipe FaceLandmarker (Tasks API >= 0.10).

Returns a rich placement descriptor for the headset:
  - bounding box over eyes/upper face
  - roll (in-plane tilt), plus full head pose (pitch, yaw, roll) via solvePnP
  - interpupillary distance (IPD) for physically-plausible scaling
  - ear anchor points for the side straps
"""
from pathlib import Path
import math
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import cv2

_MODEL_PATH = str(Path(__file__).parent / "assets" / "face_landmarker.task")

# FaceMesh landmark indices (same 478-point map in the Tasks API)
_LM = {
    "right_temple":   234,
    "left_temple":    454,
    "forehead_top":   10,
    "nose_bridge":    168,
    "nose_tip":       1,
    "chin":           152,
    "right_eye_out":  33,    # image-left eye, outer corner
    "left_eye_out":   263,   # image-right eye, outer corner
    "right_mouth":    61,
    "left_mouth":     291,
    "right_iris":     468,   # iris centre (needs refine landmarks)
    "left_iris":      473,
    "right_ear":      127,
    "left_ear":       356,
}

# Canonical 3D face model (mm-ish, arbitrary scale) for solvePnP.
# Order must match _POSE_LANDMARKS below.
_MODEL_3D = np.array([
    (0.0,    0.0,    0.0),     # nose tip
    (0.0,   -330.0, -65.0),    # chin
    (-225.0, 170.0, -135.0),   # right eye outer corner (image-left)
    (225.0,  170.0, -135.0),   # left eye outer corner  (image-right)
    (-150.0,-150.0, -125.0),   # right mouth corner
    (150.0, -150.0, -125.0),   # left mouth corner
], dtype=np.float64)

_POSE_LANDMARKS = ["nose_tip", "chin", "right_eye_out",
                   "left_eye_out", "right_mouth", "left_mouth"]


def _build_detector():
    base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def _rotation_to_euler(R):
    """Returns (pitch, yaw, roll) in degrees from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def _wrap_angle(a):
    """Normalise an angle that solvePnP may report flipped by ~180 deg."""
    if a > 90:
        a -= 180
    elif a < -90:
        a += 180
    return a


def _estimate_head_pose(pts_px, w, h):
    """
    pts_px : list of (x, y) image points matching _POSE_LANDMARKS order.
    Returns (pitch, yaw, roll) in degrees, or (0,0,0) on failure.
    """
    image_pts = np.array(pts_px, dtype=np.float64)
    focal = float(w)
    cam = np.array([[focal, 0, w / 2.0],
                    [0, focal, h / 2.0],
                    [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, image_pts, cam, dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_to_euler(R)
    return _wrap_angle(pitch), _wrap_angle(yaw), _wrap_angle(roll)


class FaceDetector:
    def __init__(self):
        self._det = _build_detector()

    def detect(self, image_bgr: np.ndarray, pose_estimator=None):
        """
        Returns (region_dict, mask_uint8) or (None, None) if no face found.

        region_dict keys:
          x1,y1,x2,y2, cx,cy, width,height,
          angle (roll from temples),
          R (3x3 rotation matrix), pose_src ('6drepnet360' | 'solvepnp'),
          pitch, yaw, roll (Euler, fallback path only),
          ipd, face_bbox, right_ear, left_ear

        MediaPipe supplies only landmark geometry. Head pose comes from
        6DRepNet360 when ``pose_estimator`` is ready, else solvePnP.
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._det.detect(mp_img)

        if not result.face_landmarks:
            return None, None

        lm = result.face_landmarks[0]
        n = len(lm)

        def px(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        rt = px(_LM["right_temple"])
        lt = px(_LM["left_temple"])
        ft = px(_LM["forehead_top"])
        nb = px(_LM["nose_bridge"])
        re = px(_LM["right_ear"])
        le = px(_LM["left_ear"])

        # --- in-plane roll from temples (robust, used for 2D fallbacks) ----
        dx = lt[0] - rt[0]
        dy = lt[1] - rt[1]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # --- full-face bbox (for the pose-net crop) ------------------------
        xs = np.fromiter((p.x for p in lm), dtype=np.float32, count=n) * w
        ys = np.fromiter((p.y for p in lm), dtype=np.float32, count=n) * h
        face_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # --- head pose: 6DRepNet360 (preferred) or solvePnP (fallback) -----
        R = None
        pose_src = "solvepnp"
        pitch = yaw = roll = 0.0
        if pose_estimator is not None and getattr(pose_estimator, "ready", False):
            try:
                R = pose_estimator.predict(image_bgr, face_bbox)
                pose_src = "6drepnet360"
            except Exception:
                R = None
        if R is None:
            pose_pts = [px(_LM[name]) for name in _POSE_LANDMARKS]
            pitch, yaw, roll = _estimate_head_pose(pose_pts, w, h)

        # --- interpupillary distance ---------------------------------------
        if n > _LM["left_iris"]:
            ri = px(_LM["right_iris"])
            li = px(_LM["left_iris"])
        else:                                   # fallback: eye outer corners
            ri = px(_LM["right_eye_out"])
            li = px(_LM["left_eye_out"])
        ipd = float(math.hypot(li[0] - ri[0], li[1] - ri[1]))

        # --- headset bounding box ------------------------------------------
        face_w = lt[0] - rt[0]
        face_h = abs(nb[1] - ft[1])
        pad_x   = int(face_w * 0.06)
        pad_top = int(face_h * 0.18)
        pad_bot = int(face_h * 0.06)

        x1 = max(0, rt[0] - pad_x)
        x2 = min(w, lt[0] + pad_x)
        y1 = max(0, ft[1] - pad_top)
        y2 = min(h, nb[1] + pad_bot)

        region = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
            "width": x2 - x1, "height": y2 - y1,
            "angle": angle,
            "R": R, "pose_src": pose_src,
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "ipd": ipd, "face_bbox": face_bbox,
            "right_ear": re, "left_ear": le,
        }

        # Soft mask (for AI inpainting mode)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        blur_k = max(3, (min(region["width"], region["height"]) // 8) | 1)
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        return region, mask

    def close(self):
        self._det.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
