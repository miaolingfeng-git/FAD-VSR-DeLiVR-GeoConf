from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def average(self):
        return self.sum / max(self.count, 1)

    def update(self, value, n: int = 1):
        if torch.is_tensor(value):
            value = value.detach().item()
        self.sum += float(value) * n
        self.count += n


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = pred[..., :, 1:] - pred[..., :, :-1]
    pred_y = pred[..., 1:, :] - pred[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def haar_subbands(x: torch.Tensor):
    """Differentiable Haar coefficients for the wavelet loss."""
    h, w = x.shape[-2:]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) / 2.0
    lh = (-x00 - x01 + x10 + x11) / 2.0
    hl = (-x00 + x01 - x10 + x11) / 2.0
    hh = (x00 - x01 - x10 + x11) / 2.0
    return ll, lh, hl, hh


def wavelet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bands = haar_subbands(pred)
    target_bands = haar_subbands(target)
    return sum(F.l1_loss(p, t) for p, t in zip(pred_bands[1:], target_bands[1:])) / 3.0


def psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    return 100.0 if mse == 0 else -10.0 * np.log10(mse / (max_value**2))


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    tensor = tensor.detach().float().clamp(0, 1).cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, epoch: int, best: float, config: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_psnr": best,
            "config": config,
        },
        path,
    )


def write_json(path: str | Path, data: Dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

