

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

    def __len__(self):
        return len(self.items)

    # -- video path (decord) -------------------------------------------------
    def _load_video(self, path):
        import decord  # imported lazily so dummy runs need no decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, width=self.img_size, height=self.img_size)
        total = len(vr)
        span = self.num_frames * self.stride
        start = 0 if total <= span else random.randint(0, total - span)
        idxs = [min(start + i * self.stride, total - 1) for i in range(self.num_frames)]
        frames = vr.get_batch(idxs).asnumpy()         # (T,H,W,C) uint8
        return frames

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