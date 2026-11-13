"""TrackNetV3 RGB preprocessing and weighted sliding-window inference.

The architecture is vendored unchanged. This small runner implements the authors'
concat-background/weighted-ensemble path without their training-only imports.
"""
from collections import deque

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .assets import TRACKNET_SHA256, sha256
from .tracknet_model import TrackNet


def device_for(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_cuda_device(requested="cuda:0"):
    """Require GPU inference instead of silently selecting a CPU backend."""
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError("Model inference requires a CUDA device, e.g. 'cuda:0'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this Python kernel; use the CUDA-enabled environment on the GPU host.")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} does not exist.")
    return torch.device("cuda", index)


def locate(heatmap, width, height):
    contours, _ = cv2.findContours((heatmap > .5).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (np.nan, np.nan, False)
    x, y, w, h = max((cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3])
    # Match the reference: truncate the heatmap center before scaling.
    return (int(int(x + w / 2) * width / 512), int(int(y + h / 2) * height / 288), True)


class TrackNetTracker:
    """Load verified weights once and reuse the model for consecutive chunks.

    Backgrounds and sliding-window ensembles remain local to each call, so
    retaining the model does not change chunk context or shuttle coordinates.
    """

    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        if sha256(checkpoint) != TRACKNET_SHA256:
            raise ValueError("Unsupported TrackNet checkpoint; use `monotrack fetch`")
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if weights["param_dict"]["seq_len"] != 8 or weights["param_dict"]["bg_mode"] != "concat":
            raise ValueError("Expected the authors' eight-frame concat-background model")
        self.model = TrackNet(27, 8).eval()
        self.model.load_state_dict(weights["model"])
        self.model.to(self.device)

    def __call__(self, frames, batch_size=1):
        if len(frames) < 8:
            raise ValueError("TrackNet requires at least eight consecutive frames")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        height, width = frames[0].shape[:2]
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
        background = np.median(rgb, axis=0).astype(np.uint8)

        def resize(image):
            # Pillow's default RGB resampling (bicubic) matches the upstream dataset.
            return np.moveaxis(np.asarray(Image.fromarray(image).resize((512, 288))), -1, 0)

        background = resize(background)
        resized = np.stack([resize(f) for f in rgb])
        del rgb
        pending = deque()
        result = []
        n_windows = len(frames) - 7
        ensemble_weights = np.array([1, 2, 3, 4, 4, 3, 2, 1], dtype=np.float32) / 20

        def emit(frame_index):
            available = [p[frame_index - start] for start, p in pending
                         if 0 <= frame_index - start < 8]
            if len(available) == 8:
                # Windows are oldest first, so their local indices run 7..0.
                heatmap = np.sum(np.stack(available) * ensemble_weights[::-1, None, None], axis=0)
            else:
                heatmap = np.mean(available, axis=0)
            result.append(locate(heatmap, width, height))

        with torch.inference_mode():
            for start in tqdm(range(0, n_windows, batch_size), desc="TrackNet windows"):
                starts = range(start, min(start + batch_size, n_windows))
                batch = np.stack([np.concatenate((background, resized[s:s + 8].reshape(24, 288, 512)))
                                  for s in starts]).astype(np.float32) / 255
                output = self.model(torch.from_numpy(batch).to(self.device)).cpu().numpy()
                for s, prediction in zip(starts, output):
                    pending.append((s, prediction))
                    while pending[0][0] < s - 7:
                        pending.popleft()
                    emit(s)
            for index in range(n_windows, len(frames)):
                emit(index)
        return result


def track(frames, checkpoint, device, batch_size=1):
    """One-shot compatibility helper; reuse TrackNetTracker for multiple chunks."""
    if len(frames) < 8:
        raise ValueError("TrackNet requires at least eight consecutive frames")
    return TrackNetTracker(checkpoint, device)(frames, batch_size=batch_size)
