#!/usr/bin/env python3
"""Standalone ROI inference example (feature extraction + MIL prediction).

This script is intentionally self-contained and follows the same two-stage flow used
in `extract_roi_features.ipynb` and `mil_roi_model_on_the_fly_lora.ipynb`:
1) extract ROI tile features with iBOT ViT (+ optional LoRA checkpoint),
2) run ROI-level MIL classification on the extracted tile features.

It uses bundled assets by default:
- ROI images: `../example_rois`
- MIL weights: `../mil_models/abmil_roi.pth`

Outputs are printed to stdout only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from rl_benchmarks.models import iBOTViT
from rl_benchmarks.models.slide_models.abmil import ABMIL
from torchvision import transforms



class IBOTMultiTaskModel(nn.Module):
    """Notebook-compatible wrapper used to load optional LoRA encoder checkpoints."""

    def __init__(self, num_classes: Sequence[int] | int, ibot_weights_path: str):
        super().__init__()
        self.base_model = iBOTViT(
            architecture="vit_base_pancan",
            encoder="teacher",
            weights_path=ibot_weights_path,
        )
        self.num_features = 768
        self.num_classes = num_classes
        if isinstance(num_classes, (list, tuple)):
            self.heads = nn.ModuleList([nn.Linear(self.num_features, int(nc)) for nc in num_classes])
        else:
            self.head = self.base_model.head
            self.head.fc = nn.Linear(self.num_features, int(num_classes))

    def forward(self, x: torch.Tensor):
        x = self.base_model(x)
        if isinstance(self.num_classes, (list, tuple)):
            return [head(x) for head in self.heads]
        return self.head(x)


def strip_prefix_if_present(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def load_torch_checkpoint(path: Path):
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise ValueError(f"Unsupported checkpoint format: {path}")


def simple_tiler(pil_img: Image.Image, tile_size: int) -> List[Image.Image]:
    w, h = pil_img.size
    tiles: List[Image.Image] = []
    for y in range(h // tile_size):
        for x in range(w // tile_size):
            x0 = x * tile_size
            y0 = y * tile_size
            tiles.append(pil_img.crop((x0, y0, x0 + tile_size, y0 + tile_size)))
    return tiles


def build_feature_extractor(
    ibot_weights_path: str,
    encoder_ckpt: Optional[Path],
    device: torch.device,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> nn.Module:
    model = IBOTMultiTaskModel(num_classes=[3, 3], ibot_weights_path=ibot_weights_path)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["qkv"],
    )
    model = get_peft_model(model, lora_config)

    if encoder_ckpt is not None:
        sd = strip_prefix_if_present(load_torch_checkpoint(encoder_ckpt), "module.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] encoder missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] encoder unexpected keys: {len(unexpected)}")

    model.eval().to(device)
    feature_extractor = model.base_model.base_model
    feature_extractor.eval().to(device)
    return feature_extractor


@torch.no_grad()
def extract_roi_features(
    image_path: Path,
    feature_extractor: nn.Module,
    device: torch.device,
    tile_size: int,
    vit_img_size: int,
    batch_size_tiles: int,
    max_tiles: Optional[int],
) -> torch.Tensor:
    img = Image.open(str(image_path)).convert("RGB")
    tiles = simple_tiler(img, tile_size=tile_size)
    if not tiles:
        raise RuntimeError(f"No tiles created from image {image_path} with tile_size={tile_size}")

    if max_tiles is not None and len(tiles) > max_tiles:
        idx = np.random.choice(len(tiles), size=max_tiles, replace=False)
        tiles = [tiles[i] for i in idx]

    trans = transforms.Compose(
        [
            transforms.Resize((vit_img_size, vit_img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    feats = []
    for i in range(0, len(tiles), batch_size_tiles):
        batch = tiles[i : i + batch_size_tiles]
        x = torch.stack([trans(t) for t in batch], dim=0).to(device)
        y = feature_extractor(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        feats.append(y.detach().float().cpu())

    return torch.cat(feats, dim=0)


@torch.no_grad()
def predict_roi(mil_model: nn.Module, roi_features: torch.Tensor, device: torch.device) -> Tuple[int, int, float, float]:
    out = mil_model(roi_features.unsqueeze(0).to(device))
    if not isinstance(out, (list, tuple)) or len(out) < 2:
        raise RuntimeError("MIL model output format is unexpected; expected [type_logits, grade_logits]")

    type_probs = F.softmax(out[0], dim=-1).squeeze(0).cpu().numpy()
    grade_probs = F.softmax(out[1], dim=-1).squeeze(0).cpu().numpy()
    type_pred = int(np.argmax(type_probs))
    grade_pred = int(np.argmax(grade_probs))
    return type_pred, grade_pred, float(type_probs[type_pred]), float(grade_probs[grade_pred])


def to_label_name(type_pred: int, grade_pred: int) -> str:
    if type_pred == 0:
        return "normal"
    if type_pred == 1:
        return f"dcis-{grade_pred + 1}"
    return f"ibc-{grade_pred + 1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--roi_dir", type=Path, default=repo_root / "example_rois")
    parser.add_argument("--ibot_weights", type=Path, required=True)
    parser.add_argument("--encoder_ckpt", type=Path, default=None)
    parser.add_argument("--mil_ckpt", type=Path, default=repo_root / "mil_models" / "abmil_roi.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tile_size", type=int, default=336)
    parser.add_argument("--vit_img_size", type=int, default=384)
    parser.add_argument("--batch_size_tiles", type=int, default=16)
    parser.add_argument("--max_tiles", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    args = parser.parse_args()

    if "cuda" in args.device.lower() and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if not args.roi_dir.exists():
        raise FileNotFoundError(f"ROI directory not found: {args.roi_dir}")
    if not args.ibot_weights.exists():
        raise FileNotFoundError(f"iBOT weights not found: {args.ibot_weights}")
    if args.encoder_ckpt is not None and not args.encoder_ckpt.exists():
        raise FileNotFoundError(f"encoder checkpoint not found: {args.encoder_ckpt}")
    if not args.mil_ckpt.exists():
        raise FileNotFoundError(f"MIL checkpoint not found: {args.mil_ckpt}")

    image_paths = sorted(
        p
        for p in args.roi_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    )
    if not image_paths:
        raise RuntimeError(f"No ROI images found in {args.roi_dir}")

    feature_extractor = build_feature_extractor(
        ibot_weights_path=str(args.ibot_weights),
        encoder_ckpt=args.encoder_ckpt,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    mil_model = ABMIL(768, [3, 3], d_model_attention=128, temperature=1.0, mlp_hidden=[128, 64]).to(device)
    mil_model.eval()
    mil_state = strip_prefix_if_present(load_torch_checkpoint(args.mil_ckpt), "module.")
    missing, unexpected = mil_model.load_state_dict(mil_state, strict=False)
    if missing:
        print(f"[WARN] MIL missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] MIL unexpected keys: {len(unexpected)}")

    print(f"[INFO] Running inference for {len(image_paths)} ROI images")
    print("image_name,pred_label,conf_type,conf_grade")

    for image_path in image_paths:
        try:
            feat = extract_roi_features(
                image_path=image_path,
                feature_extractor=feature_extractor,
                device=device,
                tile_size=args.tile_size,
                vit_img_size=args.vit_img_size,
                batch_size_tiles=args.batch_size_tiles,
                max_tiles=args.max_tiles,
            )
            type_pred, grade_pred, conf_type, conf_grade = predict_roi(mil_model, feat, device)
            pred_label = to_label_name(type_pred, grade_pred)
            print(f"{image_path.name},{pred_label},{conf_type:.4f},{conf_grade:.4f}")
        except Exception as exc:
            print(f"{image_path.name},ERROR,{exc},")


if __name__ == "__main__":
    main()
