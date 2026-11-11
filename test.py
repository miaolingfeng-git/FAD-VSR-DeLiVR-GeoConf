from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

import datasets
import models
from utils import psnr, save_image, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/test.yaml")
    parser.add_argument("--model", required=True, help="path to epoch-best.pth")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def make_device(args):
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _starts(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.no_grad()
def infer_tiled(model, clip: torch.Tensor, scale: int, tile_size: int, tile_overlap: int):
    """Run one temporal clip, optionally in overlapping LR tiles."""
    b, _, _, h, w = clip.shape
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        return model(clip)
    stride = max(1, tile_size - tile_overlap)
    ys = _starts(h, tile_size, stride)
    xs = _starts(w, tile_size, stride)
    output = torch.zeros(b, 3, h * scale, w * scale, device=clip.device)
    weight = torch.zeros_like(output)
    for y in ys:
        for x in xs:
            tile = clip[..., y : min(y + tile_size, h), x : min(x + tile_size, w)]
            pred = model(tile)
            th, tw = tile.shape[-2:]
            output[..., y * scale : (y + th) * scale, x * scale : (x + tw) * scale] += pred
            weight[..., y * scale : (y + th) * scale, x * scale : (x + tw) * scale] += 1.0
    return output / weight.clamp_min(1.0)


@torch.no_grad()
def infer_video(model, lq: torch.Tensor, scale: int, num_frames: int, tile_size: int, tile_overlap: int):
    radius = num_frames // 2
    predictions = []
    for center in range(lq.shape[0]):
        indices = []
        for index in range(center - radius, center + radius + 1):
            value = index
            while value < 0 or value >= lq.shape[0]:
                if value < 0:
                    value = -value
                if value >= lq.shape[0]:
                    value = 2 * lq.shape[0] - value - 2 if lq.shape[0] > 1 else 0
            indices.append(value)
        clip = lq[indices].unsqueeze(0)
        predictions.append(infer_tiled(model, clip, scale, tile_size, tile_overlap)[0].cpu())
    return torch.stack(predictions)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed_everything(int(config.get("seed", 7)))
    device = make_device(args)
    checkpoint = torch.load(args.model, map_location="cpu")
    model_spec = checkpoint.get("config", {}).get("model", config["model"])
    model = models.make(model_spec)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.to(device).eval()

    dataset = datasets.make(config["test_dataset"])
    test_cfg = config.get("inference", {})
    scale = int(model.scale)
    output_root = Path(args.output or config.get("output_dir", "results"))
    output_root.mkdir(parents=True, exist_ok=True)
    psnr_values = []
    for item in tqdm(dataset, desc="videos"):
        name = item["name"]
        lq = item["lq"].to(device)
        pred = infer_video(
            model,
            lq,
            scale,
            model.num_frames,
            int(test_cfg.get("tile_size", 0)),
            int(test_cfg.get("tile_overlap", 16)),
        )
        save_dir = output_root / name
        save_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(pred):
            save_image(frame, save_dir / f"{index:04d}.png")
        if item["gt"] is not None:
            gt = item["gt"]
            if gt.shape[0] != pred.shape[0]:
                raise ValueError(f"GT frame count mismatch for {name}")
            values = [psnr(pred[i].unsqueeze(0), gt[i].unsqueeze(0)) for i in range(pred.shape[0])]
            value = sum(values) / len(values)
            psnr_values.append(value)
            print(f"{name}: PSNR {value:.4f} dB")
    if psnr_values:
        print(f"Average PSNR: {sum(psnr_values) / len(psnr_values):.4f} dB")


if __name__ == "__main__":
    main()

