"""
Head-pose estimation with 6DRepNet360 (Hempel et al.).

This network regresses a continuous 6D rotation representation and converts it
to a full 3x3 rotation matrix — far more robust on turned / tilted faces than
solvePnP on a handful of landmarks.

Architecture and the 6D->matrix conversion are reproduced from the official
repo (github.com/thohemp/6DRepNet360, sixdrepnet360/test.py + utils.py):
the backbone is torchvision ResNet-50 (Bottleneck, [3,4,6,3]) with a linear
head regressing 6 numbers.

We only return the rotation matrix; the overlay maps it straight into the
perspective warp. MediaPipe is kept solely for landmark geometry.
"""
from pathlib import Path
import threading
import numpy as np

_WEIGHTS = Path(__file__).parent / "assets" / "6DRepNet360.pth"

# torch is heavy / optional — import lazily so the overlay pipeline still runs
try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
    from PIL import Image
    _TORCH_OK = True
except Exception:                                   # pragma: no cover
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# 6D -> rotation matrix (verbatim math from the repo, CPU-clean)
# ---------------------------------------------------------------------------

def _normalize_vector(v):
    v_mag = torch.sqrt((v ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-8)
    return v / v_mag


def _cross_product(u, v):
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return torch.stack((i, j, k), dim=1)


def _ortho6d_to_matrix(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = _normalize_vector(x_raw)
    z = _normalize_vector(_cross_product(x, y_raw))
    y = _cross_product(z, x)
    return torch.stack((x, y, z), dim=2)            # (B, 3, 3)


# ---------------------------------------------------------------------------
# model (ResNet-50 backbone + 6D regression head)
# ---------------------------------------------------------------------------

if _TORCH_OK:

    class SixDRepNet360(nn.Module):
        def __init__(self, block, layers):
            self.inplanes = 64
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.linear_reg = nn.Linear(512 * block.expansion, 6)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x).view(x.size(0), -1)
            return _ortho6d_to_matrix(self.linear_reg(x))


class HeadPoseEstimator:
    """
    Lazy, CPU-only 6DRepNet360 wrapper.

    estimator = HeadPoseEstimator()
    if estimator.ready:
        R = estimator.predict(image_bgr, (x1, y1, x2, y2))   # 3x3 np.float32
    """

    def __init__(self):
        self.ready = False
        self._model = None
        self._tf = None
        self._lock = threading.Lock()
        if not _TORCH_OK or not _WEIGHTS.exists():
            return
        try:
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
            model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
            sd = torch.load(str(_WEIGHTS), map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            model.load_state_dict(sd, strict=True)
            model.eval()
            self._model = model
            self._tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.ready = True
        except Exception as exc:                    # pragma: no cover
            print(f"[head_pose] 6DRepNet360 unavailable, falling back: {exc}")
            self.ready = False

    def predict(self, image_bgr, bbox, pad=0.25):
        """Returns a 3x3 float32 rotation matrix for the face in bbox."""
        if not self.ready:
            raise RuntimeError("HeadPoseEstimator not ready")
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        x1 = int(max(0, x1 - pad * bw))
        x2 = int(min(w, x2 + pad * bw))
        y1 = int(max(0, y1 - pad * bh))
        y2 = int(min(h, y2 + pad * bh))
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise RuntimeError("empty face crop")

        rgb = crop[:, :, ::-1]                       # BGR -> RGB
        pil = Image.fromarray(np.ascontiguousarray(rgb))
        tens = self._tf(pil).unsqueeze(0)
        with self._lock, torch.no_grad():
            R = self._model(tens)[0].cpu().numpy().astype(np.float32)
        return R
