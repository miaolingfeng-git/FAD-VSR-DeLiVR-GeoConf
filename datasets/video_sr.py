"""Paired video super-resolution datasets.

Expected layout:

    root/
      lq/<video_name>/*.png
      gt/<video_name>/*.png

The dataset also accepts ``root/lq`` and ``root/gt`` directly through the
configuration file.  Frames are paired by video directory name and sorted by
filename.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _read_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _frame_paths(directory: Path) -> List[Path]:
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def _video_dirs(root: Path) -> Dict[str, List[Path]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if dirs:
        result = {p.name: _frame_paths(p) for p in dirs}
        return {k: v for k, v in result.items() if v}
    frames = _frame_paths(root)
    return {root.name: frames} if frames else {}


def _paired_videos(lq_root: Path, gt_root: Optional[Path]) -> List[Tuple[str, List[Path], Optional[List[Path]]]]:
    lq = _video_dirs(lq_root)
    if not lq:
        raise RuntimeError(f"no image sequences found under {lq_root}")
    if gt_root is None:
        return [(name, paths, None) for name, paths in sorted(lq.items())]
    gt = _video_dirs(gt_root)
    names = sorted(set(lq) & set(gt))
    if not names:
        raise RuntimeError(f"no matching video names between {lq_root} and {gt_root}")
    pairs = []
    for name in names:
        if len(lq[name]) != len(gt[name]):
            raise ValueError(f"frame count mismatch for {name}: {len(lq[name])} vs {len(gt[name])}")
        pairs.append((name, lq[name], gt[name]))
    return pairs


def _reflect_indices(length: int, center: int, radius: int) -> List[int]:
    indices = []
    for index in range(center - radius, center + radius + 1):
        value = index
        while value < 0 or value >= length:
            if value < 0:
                value = -value
            if value >= length:
                value = 2 * length - value - 2 if length > 1 else 0
        indices.append(value)
    return indices


def _paired_crop(lq: torch.Tensor, gt: torch.Tensor, patch_size: int, scale: int):
    _, h, w = lq.shape
    if h < patch_size or w < patch_size:
        pad_h, pad_w = max(0, patch_size - h), max(0, patch_size - w)
        lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="replicate")
        gt = torch.nn.functional.pad(gt, (0, pad_w * scale, 0, pad_h * scale), mode="replicate")
        _, h, w = lq.shape
    top = random.randint(0, h - patch_size) if h > patch_size else 0
    left = random.randint(0, w - patch_size) if w > patch_size else 0
    lq = lq[:, top : top + patch_size, left : left + patch_size]
    gt = gt[:, top * scale : (top + patch_size) * scale, left * scale : (left + patch_size) * scale]
    return lq, gt


class PairedVideoDataset(Dataset):
    """Random temporal windows for training or deterministic windows for validation."""

    def __init__(
        self,
        lq_root: str,
        gt_root: str,
        num_frames: int = 7,
        scale: int = 4,
        patch_size: int = 64,
        train: bool = True,
        samples_per_video: int = 100,
        augment: bool = True,
    ):
        if num_frames % 2 != 1:
            raise ValueError("num_frames must be odd")
        self.videos = _paired_videos(Path(lq_root), Path(gt_root))
        self.num_frames = num_frames
        self.radius = num_frames // 2
        self.scale = scale
        self.patch_size = patch_size
        self.train = train
        self.augment = augment and train
        self.samples_per_video = max(1, samples_per_video)

    def __len__(self):
        if self.train:
            return len(self.videos) * self.samples_per_video
        return len(self.videos)

    def _sample_center(self, frames: int, index: int) -> int:
        if self.train:
            return random.randrange(frames)
        return frames // 2 if frames else 0

    def __getitem__(self, index: int):
        video_index = index % len(self.videos) if self.train else index
        name, lq_paths, gt_paths = self.videos[video_index]
        if gt_paths is None:
            raise RuntimeError("PairedVideoDataset requires gt_root")
        center = self._sample_center(len(lq_paths), index)
        indices = _reflect_indices(len(lq_paths), center, self.radius)
        lq_frames = torch.stack([_read_image(lq_paths[i]) for i in indices])
        gt = _read_image(gt_paths[center])
        if self.train:
            lq_frames, gt = self._crop_clip(lq_frames, gt)
            if self.augment:
                if random.random() < 0.5:
                    lq_frames = lq_frames.flip(-1)
                    gt = gt.flip(-1)
                if random.random() < 0.5:
                    lq_frames = lq_frames.flip(-2)
                    gt = gt.flip(-2)
                if random.random() < 0.5:
                    lq_frames = lq_frames.flip(0)
        else:
            lq_frames, gt = self._crop_clip(lq_frames, gt, deterministic=True)
        return {"lq": lq_frames, "gt": gt, "name": name, "center": center}

    def _crop_clip(self, lq_frames: torch.Tensor, gt: torch.Tensor, deterministic: bool = False):
        _, _, h, w = lq_frames.shape
        patch = self.patch_size
        if h < patch or w < patch:
            pad_h, pad_w = max(0, patch - h), max(0, patch - w)
            lq_frames = torch.nn.functional.pad(lq_frames, (0, pad_w, 0, pad_h), mode="replicate")
            gt = torch.nn.functional.pad(gt, (0, pad_w * self.scale, 0, pad_h * self.scale), mode="replicate")
            _, _, h, w = lq_frames.shape
        top = (h - patch) // 2 if deterministic else random.randint(0, h - patch)
        left = (w - patch) // 2 if deterministic else random.randint(0, w - patch)
        lq_frames = lq_frames[:, :, top : top + patch, left : left + patch]
        gt = gt[:, top * self.scale : (top + patch) * self.scale, left * self.scale : (left + patch) * self.scale]
        return lq_frames, gt


class VimeoSeptupletDataset(Dataset):
    """Vimeo-90K septuplet dataset using official sequence manifests.

    Each item is one 7-frame sequence. During training, the dataset length is
    an explicit number of random samples so that optimization is iteration-
    based rather than tied to an impractically large ``epoch`` over all
    septuplets. The official test split is never used for training.
    """

    def __init__(
        self,
        lq_root: str,
        gt_root: str,
        manifest: str,
        num_frames: int = 7,
        scale: int = 4,
        patch_size: int = 64,
        train: bool = True,
        samples_per_epoch: int = 10000,
        augment: bool = True,
    ):
        if num_frames != 7:
            raise ValueError("Vimeo-90K septuplet training requires num_frames=7")
        self.lq_root = Path(lq_root)
        self.gt_root = Path(gt_root)
        self.manifest = Path(manifest)
        self.scale = int(scale)
        self.patch_size = int(patch_size)
        self.train = bool(train)
        self.augment = bool(augment and train)
        self.samples_per_epoch = max(1, int(samples_per_epoch))
        if not self.lq_root.exists():
            raise FileNotFoundError(f"Vimeo LQ root does not exist: {self.lq_root}")
        if not self.gt_root.exists():
            raise FileNotFoundError(f"Vimeo GT root does not exist: {self.gt_root}")
        if not self.manifest.exists():
            raise FileNotFoundError(f"Vimeo manifest does not exist: {self.manifest}")
        self.sequences = [
            line.strip().split()[0]
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not self.sequences:
            raise RuntimeError(f"empty Vimeo manifest: {self.manifest}")

    def __len__(self):
        return self.samples_per_epoch if self.train else len(self.sequences)

    @staticmethod
    def _sequence_root(root: Path, rel: str) -> Path:
        # Accept either .../sequences as root or the dataset root.
        direct = root / rel
        nested = root / "sequences" / rel
        return direct if direct.exists() else nested

    def _paths(self, rel: str):
        lq_dir = self._sequence_root(self.lq_root, rel)
        gt_dir = self._sequence_root(self.gt_root, rel)
        lq = [lq_dir / f"im{i}.png" for i in range(1, 8)]
        gt = [gt_dir / f"im{i}.png" for i in range(1, 8)]
        if not all(p.exists() for p in lq + gt):
            missing = next(p for p in lq + gt if not p.exists())
            raise FileNotFoundError(f"missing Vimeo frame: {missing}")
        return lq, gt

    def __getitem__(self, index: int):
        seq_index = random.randrange(len(self.sequences)) if self.train else index
        rel = self.sequences[seq_index]
        lq_paths, gt_paths = self._paths(rel)
        lq_frames = torch.stack([_read_image(p) for p in lq_paths])
        gt = _read_image(gt_paths[3])
        lq_frames, gt = self._crop_clip(lq_frames, gt, deterministic=not self.train)
        if self.augment:
            if random.random() < 0.5:
                lq_frames = lq_frames.flip(-1); gt = gt.flip(-1)
            if random.random() < 0.5:
                lq_frames = lq_frames.flip(-2); gt = gt.flip(-2)
            if random.random() < 0.5:
                lq_frames = lq_frames.flip(0)
        return {"lq": lq_frames, "gt": gt, "name": rel, "center": 3}

    def _crop_clip(self, lq_frames, gt, deterministic=False):
        _, _, h, w = lq_frames.shape
        patch = self.patch_size
        if h < patch or w < patch:
            pad_h, pad_w = max(0, patch - h), max(0, patch - w)
            lq_frames = torch.nn.functional.pad(lq_frames, (0, pad_w, 0, pad_h), mode="replicate")
            gt = torch.nn.functional.pad(gt, (0, pad_w * self.scale, 0, pad_h * self.scale), mode="replicate")
            _, _, h, w = lq_frames.shape
        top = (h - patch) // 2 if deterministic else random.randint(0, h - patch)
        left = (w - patch) // 2 if deterministic else random.randint(0, w - patch)
        lq_frames = lq_frames[:, :, top:top + patch, left:left + patch]
        gt = gt[:, top * self.scale:(top + patch) * self.scale,
                left * self.scale:(left + patch) * self.scale]
        return lq_frames, gt


class VimeoSeptupletFolder(Dataset):
    """Full official Vimeo-90K sequences for evaluation."""

    def __init__(self, lq_root: str, gt_root: str, manifest: str):
        self.lq_root = Path(lq_root)
        self.gt_root = Path(gt_root)
        self.sequences = [
            line.strip().split()[0]
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not self.sequences:
            raise RuntimeError(f"empty Vimeo manifest: {manifest}")

    @staticmethod
    def _sequence_root(root: Path, rel: str) -> Path:
        direct = root / rel
        nested = root / "sequences" / rel
        return direct if direct.exists() else nested

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int):
        rel = self.sequences[index]
        lq_dir = self._sequence_root(self.lq_root, rel)
        gt_dir = self._sequence_root(self.gt_root, rel)
        lq = torch.stack([_read_image(lq_dir / f"im{i}.png") for i in range(1, 8)])
        gt = torch.stack([_read_image(gt_dir / f"im{i}.png") for i in range(1, 8)])
        return {"name": rel, "lq": lq, "gt": gt}


class VideoPairFolder(Dataset):
    """Full sequences for inference and metric evaluation."""

    def __init__(self, lq_root: str, gt_root: Optional[str] = None):
        self.videos = _paired_videos(Path(lq_root), Path(gt_root) if gt_root else None)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index: int):
        name, lq_paths, gt_paths = self.videos[index]
        lq = torch.stack([_read_image(path) for path in lq_paths])
        gt = torch.stack([_read_image(path) for path in gt_paths]) if gt_paths is not None else None
        return {"name": name, "lq": lq, "gt": gt}


VideoSRDataset = PairedVideoDataset

