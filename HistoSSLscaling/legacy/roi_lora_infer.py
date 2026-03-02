#!/usr/bin/env python3
"""
ROI inference script (BCaCAD-style)

- Loads a (LoRA-finetuned) iBOT ViT encoder checkpoint (optional) to extract tile features per ROI image
- Loads a MIL classifier checkpoint (e.g., ABMIL) to predict:
    head0: tumor type  (Normal / nonIBC / IBC)  -> 3 classes
    head1: grade       (Low / Medium / High)    -> 3 classes
- Saves per-ROI predictions to CSV/JSON (ONLY successful samples; runtime-error samples are skipped)
- Samples mispredicted ROI images and exports:
    - a mispredictions CSV
    - copied/thumbnails of mispredicted ROIs (filename includes row_key + TRUE/PRED labels)
    - a montage PNG for quick inspection

Example:
python roi_infer.py \
  --roi_root /mnt/hd0/project/bcacad/data/roi-level/suqh/model \
  --ibot_weights /mnt/hd1/bcacad/ibot_vit_base_pancan.pth \
  --encoder_ckpt /mnt/hd1/bcacad/timm_lora_ft/2025_07_14_04_18_54/model-13.pth \
  --mil_ckpt /mnt/hd1/bcacad/mil_ckpts/abmil_epoch10.pth \
  --out_dir /mnt/hd1/bcacad/infer_runs/2026_02_21

Notes:
- This script tries to import QiLuROI (your ROI tiler). If unavailable, it falls back to a simple non-overlapping tiler.
- Your environment must have:
    rl_benchmarks, peft, timm, torch, torchvision, PIL
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# --- rl_benchmarks models ---
from rl_benchmarks.models import iBOTViT
from rl_benchmarks.models.slide_models.abmil import ABMIL
from rl_benchmarks.models.slide_models.chowder import Chowder
from rl_benchmarks.models.slide_models.dsmil import DSMIL
from rl_benchmarks.models.slide_models.hiptmil import HIPTMIL
from rl_benchmarks.models.slide_models.meanpool import MeanPool
from rl_benchmarks.models.slide_models.transmil import TransMIL

# --- peft / LoRA ---
from peft import LoraConfig, get_peft_model, TaskType

# ----------------------------
# Defaults (label mapping)
# ----------------------------
DEFAULT_FOLDER_TO_LABEL = {
    "normal": [0, 0],
    "dcis-1": [1, 0],
    "dcis-2": [1, 1],
    "dcis-3": [1, 2],
    "ibc-1": [2, 0],
    "ibc-2": [2, 1],
    "ibc-3": [2, 2],
}
TYPE_NAMES = ["Normal", "nonIBC", "IBC"]
GRADE_NAMES = ["Low", "Medium", "High"]


# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(s: str, max_len: int = 180) -> str:
    """
    Make a filesystem-safe filename component.
    Keeps letters/numbers/._- and replaces everything else with "_".
    """
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s[:max_len] if len(s) > max_len else s


def make_row_key(image_path: Path, root: Optional[Path] = None) -> str:
    """
    Stable-ish unique key for joining CSV rows <-> files.
    Uses a short SHA1 hash + readable tail (relative path if root is provided).
    """
    try:
        if root is not None:
            rel = image_path.resolve().relative_to(root.resolve())
            rel_str = rel.as_posix()
        else:
            rel_str = image_path.resolve().as_posix()
    except Exception:
        rel_str = str(image_path)

    h = hashlib.sha1(rel_str.encode("utf-8")).hexdigest()[:10]
    tail = safe_filename(rel_str.replace("/", "__"))
    return f"{h}__{tail}"


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def load_torch_checkpoint(path: Path) -> Dict:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise ValueError(f"Unsupported checkpoint format at: {path}")


def make_label_name(type_idx: int, grade_idx: int) -> str:
    if type_idx == 0:
        return "normal"
    if type_idx == 1:
        return f"dcis-{grade_idx+1}"
    if type_idx == 2:
        return f"ibc-{grade_idx+1}"
    return "unknown"


# ----------------------------
# ROI tiling (QiLuROI fallback)
# ----------------------------
def try_import_qiluroi():
    try:
        from patch_based_test.img import QiLuROI  # type: ignore
        return QiLuROI
    except Exception:
        return None


def simple_tiler(pil_img: Image.Image, tile_size: int) -> List[Image.Image]:
    """Non-overlapping tiles; drops incomplete border tiles."""
    w, h = pil_img.size
    tiles: List[Image.Image] = []
    nx = w // tile_size
    ny = h // tile_size
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_size
            y0 = iy * tile_size
            tile = pil_img.crop((x0, y0, x0 + tile_size, y0 + tile_size))
            tiles.append(tile)
    return tiles


def get_tiles(image_path: Path, tile_size: int, src_mag: int, tar_mag: int) -> List[Image.Image]:
    QiLuROI = try_import_qiluroi()
    if QiLuROI is not None:
        im = QiLuROI(str(image_path), src_mag, tar_mag, tile_size)
        im.setIterator(tile_size)
        return [p for p in im]
    pil = Image.open(str(image_path)).convert("RGB")
    return simple_tiler(pil, tile_size)


# ----------------------------
# Encoder (iBOT + LoRA)
# ----------------------------
class IBOTMultiTaskModel(nn.Module):
    """
    Same shape as the notebook:
      - iBOTViT backbone
      - optional multi-head (kept for state_dict compatibility)
    """
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


def build_feature_extractor(
    ibot_weights_path: str,
    encoder_ckpt: Optional[Path],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
    device: torch.device,
) -> nn.Module:
    # Keep num_classes=[3,3] for checkpoint compatibility; not used in feature extraction.
    model = IBOTMultiTaskModel(num_classes=[3, 3], ibot_weights_path=ibot_weights_path)

    # Match notebook LoRA style (FEATURE_EXTRACTION + target qkv)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
    )
    model = get_peft_model(model, lora_config)

    if encoder_ckpt is not None:
        sd = load_torch_checkpoint(encoder_ckpt)
        sd = strip_prefix_if_present(sd, "module.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] encoder missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
        if unexpected:
            print(f"[WARN] encoder unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")

    model.eval().to(device)

    # Return the underlying timm encoder for feature extraction.
    feature_extractor = model.base_model.base_model
    feature_extractor.eval().to(device)
    return feature_extractor


# ----------------------------
# MIL classifier
# ----------------------------
def get_mil_model(model_name: str, in_dim: int, out_dim: Sequence[int] | int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "abmil":
        return ABMIL(in_dim, out_dim, d_model_attention=128, temperature=1.0, mlp_hidden=[128, 64])
    if model_name == "chowder":
        return Chowder(in_dim, out_dim, n_top=2, n_bottom=2, tiles_mlp_hidden=[128], mlp_hidden=[128, 64])
    if model_name == "dsmil":
        return DSMIL(
            in_dim,
            out_dim,
            d_tiles_values=32,
            d_tiles_queries=32,
            passing_values=False,
            tiles_scores_mlp_hidden=[200, 100],
            tiles_queries_mlp_hidden=[200, 100],
            mlp_hidden=[200, 100],
        )
    if model_name == "hiptmil":
        return HIPTMIL(in_dim, out_dim)
    if model_name == "transmil":
        return TransMIL(in_dim, out_features=out_dim)
    if model_name == "meanpool":
        return MeanPool(in_dim, out_dim)
    raise ValueError(f"Unknown mil_arch: {model_name}")


def load_mil_weights(model: nn.Module, mil_ckpt: Path) -> None:
    sd = load_torch_checkpoint(mil_ckpt)
    sd = strip_prefix_if_present(sd, "module.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] MIL missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] MIL unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")


# ----------------------------
# Data discovery
# ----------------------------
@dataclass
class ROIItem:
    image_path: Path
    true_type: int
    true_grade: int
    true_label_name: str


def load_label_map(label_map_json: Optional[Path]) -> Dict[str, List[int]]:
    if label_map_json is None:
        return dict(DEFAULT_FOLDER_TO_LABEL)
    with open(label_map_json, "r") as f:
        d = json.load(f)
    out: Dict[str, List[int]] = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[str(k)] = [int(v[0]), int(v[1])]
        else:
            raise ValueError(f"label_map_json value must be length-2 list. Bad key={k}, value={v}")
    return out


def discover_rois(
    roi_root: Optional[Path],
    input_csv: Optional[Path],
    label_map: Dict[str, List[int]],
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"),
) -> List[ROIItem]:
    items: List[ROIItem] = []
    if input_csv is not None:
        import pandas as pd
        df = pd.read_csv(input_csv)
        if "image_path" not in df.columns:
            raise ValueError("input_csv must contain column: image_path")

        for _, row in df.iterrows():
            p = Path(str(row["image_path"]))
            if "label_name" in df.columns and isinstance(row["label_name"], str):
                lname = row["label_name"]
                if lname not in label_map:
                    raise ValueError(f"label_name '{lname}' not in label_map")
                tt, gg = label_map[lname]
            elif "true_type" in df.columns and "true_grade" in df.columns:
                tt, gg = int(row["true_type"]), int(row["true_grade"])
                lname = make_label_name(tt, gg)
            else:
                lname = p.parent.name
                if lname not in label_map:
                    raise ValueError(
                        "Cannot infer label. Provide label_name column, or true_type/true_grade, "
                        f"or ensure parent folder names are in label_map. Missing: {lname}"
                    )
                tt, gg = label_map[lname]
            items.append(ROIItem(p, tt, gg, lname))
        return items

    if roi_root is None:
        raise ValueError("Provide either --roi_root or --input_csv")

    roi_root = roi_root.resolve()
    paths: List[Path] = []
    for ext in exts:
        paths.extend(list(roi_root.rglob(f"*{ext}")))
    paths = sorted(set(paths))

    for p in paths:
        lname = p.parent.name
        if lname not in label_map:
            continue
        tt, gg = label_map[lname]
        items.append(ROIItem(p, tt, gg, lname))
    return items


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def extract_features_for_roi(
    image_path: Path,
    feature_extractor: nn.Module,
    tile_size: int,
    vit_img_size: int,
    src_mag: int,
    tar_mag: int,
    batch_size_tiles: int,
    max_tiles: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    tiles = get_tiles(image_path, tile_size=tile_size, src_mag=src_mag, tar_mag=tar_mag)
    if len(tiles) == 0:
        raise RuntimeError(f"No tiles produced for ROI: {image_path}")

    if max_tiles is not None and len(tiles) > max_tiles:
        idx = np.random.choice(len(tiles), size=max_tiles, replace=False)
        tiles = [tiles[i] for i in idx]

    trans = transforms.Compose([
        transforms.Resize((vit_img_size, vit_img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats: List[torch.Tensor] = []
    for i in range(0, len(tiles), batch_size_tiles):
        batch = tiles[i:i + batch_size_tiles]
        x = torch.stack([trans(t) for t in batch], dim=0).to(device, non_blocking=True)
        y = feature_extractor(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        if not torch.is_tensor(y):
            raise RuntimeError(f"Unexpected encoder output type: {type(y)}")
        feats.append(y.detach().float().cpu())

    feat = torch.cat(feats, dim=0)  # [N, D]
    return feat


@torch.no_grad()
def predict_roi(
    mil_model: nn.Module,
    roi_features: torch.Tensor,  # [N, D] on CPU
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    x = roi_features.unsqueeze(0).to(device)  # [1, N, D]
    out = mil_model(x)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        type_logits, grade_logits = out[0], out[1]
    else:
        raise RuntimeError("MIL model must return a list/tuple [type_logits, grade_logits].")

    type_probs = F.softmax(type_logits, dim=-1).squeeze(0).detach().cpu().numpy()
    grade_probs = F.softmax(grade_logits, dim=-1).squeeze(0).detach().cpu().numpy()
    type_pred = int(np.argmax(type_probs))
    grade_pred = int(np.argmax(grade_probs))
    return type_probs, grade_probs, type_pred, grade_pred


# ----------------------------
# Mispred export helpers
# ----------------------------
def get_default_font(size: int = 16):
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_caption(img: Image.Image, caption: str, pad: int = 6, font_size: int = 16) -> Image.Image:
    font = get_default_font(font_size)
    w, h = img.size
    lines = caption.split("\n")
    lh = font_size + 4
    cap_h = pad * 2 + lh * len(lines)
    canvas = Image.new("RGB", (w, h + cap_h), (0, 0, 0))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = h + pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255), font=font)
        y += lh
    return canvas


def make_montage(
    image_paths: List[Path],
    captions: List[str],
    out_path: Path,
    thumb_size: int = 384,
    cols: int = 4,
) -> None:
    assert len(image_paths) == len(captions)
    if len(image_paths) == 0:
        return

    thumbs: List[Image.Image] = []
    for p, cap in zip(image_paths, captions):
        im = Image.open(str(p)).convert("RGB")
        im.thumbnail((thumb_size, thumb_size))
        im = draw_caption(im, cap, font_size=14)
        thumbs.append(im)

    cols = max(1, cols)
    rows = math.ceil(len(thumbs) / cols)
    cell_w = max(t.size[0] for t in thumbs)
    cell_h = max(t.size[1] for t in thumbs)

    montage = Image.new("RGB", (cell_w * cols, cell_h * rows), (30, 30, 30))
    for idx, t in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        montage.paste(t, (c * cell_w, r * cell_h))
    montage.save(str(out_path))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_root", type=str, default=None,
                    help="Root dir containing ROI images in class-named folders (e.g., dcis-2/xxx.png).")
    ap.add_argument("--input_csv", type=str, default=None,
                    help="Optional CSV with column image_path and (label_name) or (true_type,true_grade).")
    ap.add_argument("--label_map_json", type=str, default=None,
                    help="Optional JSON mapping folder/label_name -> [true_type,true_grade].")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for predictions and mispred samples.")
    ap.add_argument("--device", type=str, default="cuda:0")

    # Encoder
    ap.add_argument("--ibot_weights", type=str, required=True,
                    help="Path to iBOT base weights (e.g., ibot_vit_base_pancan.pth).")
    ap.add_argument("--encoder_ckpt", type=str, default=None,
                    help="Optional LoRA-finetuned encoder checkpoint (torch .pth).")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.2)
    ap.add_argument("--lora_target_modules", type=str, default="qkv",
                    help="Comma-separated (default: qkv).")

    # Tiling / encoding
    ap.add_argument("--tile_size", type=int, default=336,
                    help="Tile size extracted from ROI before resize.")
    ap.add_argument("--vit_img_size", type=int, default=384,
                    help="Encoder input resize.")
    ap.add_argument("--src_mag", type=int, default=10)
    ap.add_argument("--tar_mag", type=int, default=10)
    ap.add_argument("--batch_size_tiles", type=int, default=16)
    ap.add_argument("--max_tiles", type=int, default=None,
                    help="Optional cap of number of tiles sampled per ROI for speed.")

    # MIL
    ap.add_argument("--mil_arch", type=str, default="abmil",
                    choices=["abmil", "chowder", "dsmil", "hiptmil", "transmil", "meanpool"])
    ap.add_argument("--mil_ckpt", type=str, required=True,
                    help="MIL classifier checkpoint.")
    ap.add_argument("--in_dim", type=int, default=768)
    ap.add_argument("--out_dim", type=str, default="3,3",
                    help="Comma-separated output dims for heads, e.g. '3,3'.")

    # Mispred sampling
    ap.add_argument("--num_mispreds", type=int, default=24)
    ap.add_argument("--mispred_mode", type=str, default="final",
                    choices=["final", "type", "grade"])
    ap.add_argument("--mispred_strategy", type=str, default="high_conf",
                    choices=["high_conf", "random"])
    ap.add_argument("--montage_cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)
    safe_mkdir(out_dir / "mispred_samples")

    # device handling
    if "cuda" in args.device.lower() and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] CUDA requested but not available. Using CPU.")
    else:
        device = torch.device(args.device)

    # used for row_key to be consistent across runs when roi_root is provided
    roi_root_for_key = Path(args.roi_root).resolve() if args.roi_root else None

    label_map = load_label_map(Path(args.label_map_json) if args.label_map_json else None)

    items = discover_rois(
        roi_root=Path(args.roi_root) if args.roi_root else None,
        input_csv=Path(args.input_csv) if args.input_csv else None,
        label_map=label_map,
    )
    if len(items) == 0:
        raise RuntimeError("No ROI images found. Check --roi_root / --input_csv and label mapping.")
    print(f"[INFO] Found {len(items)} ROI images")

    # Build feature extractor
    feature_extractor = build_feature_extractor(
        ibot_weights_path=args.ibot_weights,
        encoder_ckpt=Path(args.encoder_ckpt) if args.encoder_ckpt else None,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[s.strip() for s in args.lora_target_modules.split(",") if s.strip()],
        device=device,
    )

    # Build MIL model
    out_dim = [int(x) for x in args.out_dim.split(",")] if "," in args.out_dim else int(args.out_dim)
    mil_model = get_mil_model(args.mil_arch, args.in_dim, out_dim).to(device)
    mil_model.eval()
    load_mil_weights(mil_model, Path(args.mil_ckpt))

    # Inference loop (skip runtime-error samples entirely)
    rows: List[Dict] = []
    mispred_rows: List[Dict] = []
    errors: List[Dict] = []

    for idx, it in enumerate(items):
        row_key = make_row_key(it.image_path, root=roi_root_for_key)

        try:
            feat = extract_features_for_roi(
                image_path=it.image_path,
                feature_extractor=feature_extractor,
                tile_size=args.tile_size,
                vit_img_size=args.vit_img_size,
                src_mag=args.src_mag,
                tar_mag=args.tar_mag,
                batch_size_tiles=args.batch_size_tiles,
                max_tiles=args.max_tiles,
                device=device,
            )
            type_probs, grade_probs, type_pred, grade_pred = predict_roi(mil_model, feat, device=device)

            true_type, true_grade = it.true_type, it.true_grade
            true_name = it.true_label_name
            pred_name = make_label_name(type_pred, grade_pred)

            correct_type = (type_pred == true_type)
            correct_grade = (grade_pred == true_grade)
            correct_final = (pred_name == true_name)

            conf_type = float(type_probs[type_pred])
            conf_grade = float(grade_probs[grade_pred])
            conf_final = min(conf_type, conf_grade)

            row = {
                "row_key": row_key,
                "image_path": str(it.image_path),
                "true_type": true_type,
                "true_grade": true_grade,
                "true_label_name": true_name,
                "pred_type": type_pred,
                "pred_grade": grade_pred,
                "pred_label_name": pred_name,
                "correct_type": int(correct_type),
                "correct_grade": int(correct_grade),
                "correct_final": int(correct_final),
                "conf_type": conf_type,
                "conf_grade": conf_grade,
                "conf_final": conf_final,
            }
            for k in range(3):
                row[f"type_prob_{k}"] = float(type_probs[k])
                row[f"grade_prob_{k}"] = float(grade_probs[k])

            rows.append(row)

            is_mispred = {
                "final": (not correct_final),
                "type": (not correct_type),
                "grade": (not correct_grade),
            }[args.mispred_mode]

            if is_mispred:
                mispred_rows.append(dict(row))

        except Exception as e:
            # Skip from prediction CSV/JSON AND skip any image export.
            errors.append({
                "row_key": row_key,
                "image_path": str(it.image_path),
                "true_type": it.true_type,
                "true_grade": it.true_grade,
                "true_label_name": it.true_label_name,
                "error": repr(e),
            })

        if (idx + 1) % 20 == 0:
            print(f"[INFO] processed {idx+1}/{len(items)}")

    if len(errors) > 0:
        print(f"[WARN] Skipped {len(errors)} samples due to runtime errors.")
        # Saved separately (NOT in the predictions CSV/JSON).
        err_path = out_dir / "roi_errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"[INFO] Saved error log (separate): {err_path}")

    if len(rows) == 0:
        raise RuntimeError("All samples failed. Check error log roi_errors.json for details.")

    # Save full predictions (successful only)
    pred_csv = out_dir / "roi_predictions.csv"
    pred_json = out_dir / "roi_predictions.json"

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    header = sorted(all_keys)

    with open(pred_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(pred_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"[INFO] Saved predictions: {pred_csv}")
    print(f"[INFO] Saved predictions: {pred_json}")

    # Mispreds sampling/export (successful only)
    if len(mispred_rows) == 0:
        print("[INFO] No mispredictions found (per mode) among successful samples.")
        return

    if args.mispred_strategy == "high_conf":
        mispred_rows_sorted = sorted(
            mispred_rows, key=lambda r: float(r.get("conf_final", 0.0)), reverse=True
        )
    else:
        mispred_rows_sorted = mispred_rows[:]
        random.shuffle(mispred_rows_sorted)

    chosen = mispred_rows_sorted[: max(1, min(args.num_mispreds, len(mispred_rows_sorted)))]

    mis_csv = out_dir / "mispredictions_sample.csv"
    with open(mis_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in chosen:
            w.writerow(r)
    print(f"[INFO] Saved mispred sample CSV: {mis_csv}")

    # copy + montage (filenames include row_key + TRUE/PRED)
    copied_paths: List[Path] = []
    captions: List[str] = []

    for i, r in enumerate(chosen):
        src = Path(r["image_path"])
        true_lbl = str(r.get("true_label_name", "unknown"))
        pred_lbl = str(r.get("pred_label_name", "unknown"))
        conf = float(r.get("conf_final", 0.0))

        dst_name = safe_filename(
            f"{i:04d}__{r.get('row_key','nokey')}__TRUE-{true_lbl}__PRED-{pred_lbl}__conf-{conf:.3f}"
        ) + src.suffix.lower()
        dst = out_dir / "mispred_samples" / dst_name

        try:
            shutil.copy2(src, dst)
        except Exception:
            im = Image.open(str(src)).convert("RGB")
            im.save(str(dst))

        copied_paths.append(dst)

        cap = (
            f"KEY: {r.get('row_key','')}\n"
            f"TRUE: {true_lbl}\n"
            f"PRED: {pred_lbl}\n"
            f"conf_final={conf:.3f}"
        )
        captions.append(cap)

    montage_path = out_dir / "mispred_montage.png"
    make_montage(copied_paths, captions, montage_path, thumb_size=384, cols=args.montage_cols)
    print(f"[INFO] Saved montage: {montage_path}")


if __name__ == "__main__":
    main()