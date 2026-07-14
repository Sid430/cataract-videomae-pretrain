
import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def _normalize(clip: torch.Tensor) -> torch.Tensor:
    """clip: (C,T,H,W) in [0,1] -> normalized."""
    return (clip - IMAGENET_MEAN) / IMAGENET_STD


class DummyVideoDataset(Dataset):
    """Random clips so the loop can be validated without any real video."""

    def __init__(self, length=64, num_frames=16, img_size=224, in_chans=3):
        self.length = length
        self.shape = (in_chans, num_frames, img_size, img_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clip = torch.rand(self.shape)        # [0,1)
        return _normalize(clip)


class CataractVideoDataset(Dataset):
    """Real cataract clips.

    Args:
        root: directory containing either .mp4 files or one subfolder per clip
              holding extracted frames (.jpg/.png).
        mode: "video" (decord) or "frames".
        num_frames, img_size, sampling_stride: clip sampling params.
    """

    def __init__(self, root, mode="video", num_frames=16, img_size=224,
                 sampling_stride=4, exts=(".mp4", ".avi", ".mov")):
        self.root = root
        self.mode = mode
        self.num_frames = num_frames
        self.img_size = img_size
        self.stride = sampling_stride

        if mode == "video":
            self.items = sorted(
                p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                if p.lower().endswith(exts))
        elif mode == "frames":
            self.items = sorted(
                d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
        else:
            raise ValueError(f"unknown mode: {mode}")
        if not self.items:
            raise RuntimeError(f"No clips found under {root} (mode={mode})")

        # pick a video decode backend: prefer decord (fast, used on Linux/GPU
        # box), fall back to OpenCV (works on Apple Silicon where decord has no
        # wheel). Only relevant in "video" mode.
        self.backend = None
        if mode == "video":
            try:
                import decord  # noqa: F401
                self.backend = "decord"
            except ImportError:
                try:
                    import cv2  # noqa: F401
                    self.backend = "opencv"
                except ImportError:
                    raise RuntimeError(
                        "video mode needs a decoder. Install one of:\n"
                        "  pip install eva-decord    (drop-in decord for macOS ARM)\n"
                        "  pip install opencv-python  (used automatically as fallback)")

    def __len__(self):
        return len(self.items)

    def _sample_indices(self, total):
        span = self.num_frames * self.stride
        start = 0 if total <= span else random.randint(0, total - span)
        return [min(start + i * self.stride, total - 1) for i in range(self.num_frames)]

    # -- video path (decord or opencv) --------------------------------------
    def _load_video(self, path):
        if self.backend == "decord":
            return self._load_video_decord(path)
        return self._load_video_cv2(path)

    def _load_video_decord(self, path):
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, width=self.img_size, height=self.img_size)
        idxs = self._sample_indices(len(vr))
        return vr.get_batch(idxs).asnumpy()           # (T,H,W,C) uint8

    def _load_video_cv2(self, path):
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:                                 # some codecs misreport count
            frames_all = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames_all.append(fr)
            cap.release()
            if not frames_all:
                raise RuntimeError(f"could not decode any frames from {path}")
            idxs = self._sample_indices(len(frames_all))
            picked = [frames_all[i] for i in idxs]
        else:
            idxs = self._sample_indices(total)
            picked, last = [], None
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = cap.read()
                if not ok:                             # seek/read failed -> reuse last good frame
                    if last is None:
                        continue
                    fr = last
                last = fr
                picked.append(fr)
            cap.release()
            if not picked:
                raise RuntimeError(f"could not decode frames from {path}")
            while len(picked) < self.num_frames:       # pad short clips
                picked.append(picked[-1])
        frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                             (self.img_size, self.img_size)) for f in picked]
        return np.stack(frames)                        # (T,H,W,C) uint8

    # -- frame-folder path ---------------------------------------------------
    def _load_frames(self, folder):
        from PIL import Image
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                       glob.glob(os.path.join(folder, "*.png")))
        total = len(files)
        span = self.num_frames * self.stride
        start = 0 if total <= span else random.randint(0, total - span)
        idxs = [min(start + i * self.stride, total - 1) for i in range(self.num_frames)]
        frames = []
        for i in idxs:
            img = Image.open(files[i]).convert("RGB").resize(
                (self.img_size, self.img_size))
            frames.append(np.asarray(img))
        return np.stack(frames)                       # (T,H,W,C) uint8

    def __getitem__(self, idx):
        path = self.items[idx]
        frames = self._load_video(path) if self.mode == "video" else self._load_frames(path)
        clip = torch.from_numpy(frames).float() / 255.0   # (T,H,W,C)
        clip = clip.permute(3, 0, 1, 2).contiguous()      # (C,T,H,W)
        return _normalize(clip)
