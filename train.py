from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
from utils import (
    AverageMeter,
    charbonnier,
    gradient_loss,
    load_checkpoint,
    psnr,
    save_checkpoint,
    seed_everything,
    wavelet_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--name", default="fadv_sr")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def make_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_lpips(weight: float, device):
    if weight <= 0:
        return None
    try:
        import lpips

        module = lpips.LPIPS(net="alex").eval().to(device)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module
    except ImportError:
        print("lpips is not installed; perceptual loss is disabled")
        return None


def train_one_epoch(loader, model, optimizer, scaler, device, weights, lpips_fn, epoch, amp):
    model.train()
    meters = {name: AverageMeter() for name in ("loss", "pixel", "wavelet", "edge", "lpips", "geometry")}
    progress = tqdm(loader, leave=False, desc=f"train {epoch}")
    for batch in progress:
        batch = move_batch(batch, device)
        lq, gt = batch["lq"], batch["gt"]
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            pred, aux = model(lq, return_aux=True)
            geometry = aux.get("geometry_consistency", pred.new_zeros(()))
            pixel = charbonnier(pred, gt)
            wave = wavelet_loss(pred, gt)
            edge = gradient_loss(pred, gt)
            perceptual = torch.zeros((), device=device)
            if lpips_fn is not None:
                perceptual = lpips_fn(pred.clamp(0, 1) * 2 - 1, gt * 2 - 1).mean()
            loss = (
                weights["pixel"] * pixel
                + weights["wavelet"] * wave
                + weights["edge"] * edge
                + weights["lpips"] * perceptual
                + weights["geometry"] * geometry
            )
        scaler.scale(loss).backward()
        if weights.get("grad_clip", 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), weights["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        values = {"loss": loss, "pixel": pixel, "wavelet": wave, "edge": edge, "lpips": perceptual, "geometry": geometry}
        for name, value in values.items():
            meters[name].update(value, lq.shape[0])
        progress.set_postfix(loss=f"{meters['loss'].average:.4f}")
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def validate(loader, model, device, amp):
    model.eval()
    meter = AverageMeter()
    for batch in tqdm(loader, leave=False, desc="valid"):
        batch = move_batch(batch, device)
        with autocast(enabled=amp):
            pred = model(batch["lq"])
        meter.update(psnr(pred, batch["gt"]), batch["lq"].shape[0])
    return meter.average


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed_everything(int(config.get("seed", 7)))
    device = make_device(args)
    print(f"device: {device}")

    model = models.make(config["model"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"].get("lr", 2e-4)),
        weight_decay=float(config["optimizer"].get("weight_decay", 1e-4)),
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=config.get("scheduler", {}).get("milestones", [150, 250]),
        gamma=float(config.get("scheduler", {}).get("gamma", 0.5)),
    )
    amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = GradScaler(enabled=amp)
    train_set = datasets.make(config["train_dataset"])
    val_set = datasets.make(config["val_dataset"])
    train_loader = DataLoader(
        train_set,
        batch_size=int(config.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    workers = int(config.get("num_workers", 0))
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(workers > 0),
    )
    lpips_fn = build_lpips(float(config.get("loss", {}).get("lpips", 0)), device)
    weights = {
        "pixel": float(config.get("loss", {}).get("pixel", 1.0)),
        "wavelet": float(config.get("loss", {}).get("wavelet", 0.05)),
        "edge": float(config.get("loss", {}).get("edge", 0.02)),
        "lpips": float(config.get("loss", {}).get("lpips", 0.0)),
        "geometry": float(config.get("loss", {}).get("geometry_consistency", 0.01)),
        "grad_clip": float(config.get("loss", {}).get("grad_clip", 1.0)),
    }
    save_dir = Path(config.get("save_dir", "experiments")) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_psnr = 1, -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_psnr = float(checkpoint.get("best_psnr", -1.0))

    epochs = int(config.get("epochs", 300))
    val_interval = max(1, int(config.get("val_interval", 1)))
    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_one_epoch(
            train_loader, model, optimizer, scaler, device, weights, lpips_fn, epoch, amp
        )
        val_psnr = None
        if epoch % val_interval == 0 or epoch == epochs:
            val_psnr = validate(val_loader, model, device, amp)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(save_dir / "epoch-best.pth", model, optimizer, scheduler, scaler, epoch, best_psnr, config)
        scheduler.step()
        save_checkpoint(save_dir / "epoch-last.pth", model, optimizer, scheduler, scaler, epoch, best_psnr, config)
        val_text = "skip" if val_psnr is None else f"{val_psnr:.3f}"
        print(
            f"epoch {epoch}/{epochs} | loss {train_stats['loss']:.5f} | "
            f"pixel {train_stats['pixel']:.5f} | val PSNR {val_text} | best {best_psnr:.3f}", flush=True
        )


if __name__ == "__main__":
    main()

