#!/usr/bin/env python3
"""Run ROI inference on bundled example ROIs using the bundled ROI MIL weights.

This is a thin wrapper around ``HistoSSLscaling/roi_lora_infer.py`` and mirrors
how ``mil_roi_model_on_the_fly_lora.ipynb`` performs ROI feature extraction + MIL
classification.

Example:
    python HistoSSLscaling/example_roi_inference.py \
      --ibot_weights /path/to/ibot_vit_base_pancan.pth

Optional LoRA encoder checkpoint:
    python HistoSSLscaling/example_roi_inference.py \
      --ibot_weights /path/to/ibot_vit_base_pancan.pth \
      --encoder_ckpt /path/to/model-13.pth
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def build_example_csv(example_rois_dir: Path, out_csv: Path, default_label: str) -> int:
    image_paths = sorted(
        p for p in example_rois_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No example ROI images found in: {example_rois_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label_name"])
        writer.writeheader()
        for image_path in image_paths:
            writer.writerow({"image_path": str(image_path.resolve()), "label_name": default_label})

    return len(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibot_weights", type=Path, required=True, help="Path to ibot_vit_base_pancan.pth")
    parser.add_argument("--encoder_ckpt", type=Path, default=None, help="Optional LoRA-finetuned encoder checkpoint")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: ./outputs/example_roi_infer_<timestamp>)")
    parser.add_argument("--default_label", type=str, default="normal", help="Temporary label_name for example CSV")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    example_rois_dir = repo_root / "example_rois"
    mil_ckpt = repo_root / "mil_models" / "abmil_roi.pth"
    runner_script = repo_root / "HistoSSLscaling" / "roi_lora_infer.py"

    if not runner_script.exists():
        raise FileNotFoundError(f"Cannot find runner script: {runner_script}")
    if not example_rois_dir.exists():
        raise FileNotFoundError(f"Cannot find example ROIs folder: {example_rois_dir}")
    if not mil_ckpt.exists():
        raise FileNotFoundError(f"Cannot find ROI MIL weights: {mil_ckpt}")
    if not args.ibot_weights.exists():
        raise FileNotFoundError(f"Cannot find iBOT weights: {args.ibot_weights}")
    if args.encoder_ckpt is not None and not args.encoder_ckpt.exists():
        raise FileNotFoundError(f"Cannot find encoder checkpoint: {args.encoder_ckpt}")

    out_dir = args.out_dir or repo_root / "outputs" / f"example_roi_infer_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    example_csv = out_dir / "example_input.csv"
    n_images = build_example_csv(example_rois_dir, example_csv, args.default_label)

    cmd = [
        sys.executable,
        str(runner_script),
        "--input_csv", str(example_csv),
        "--mil_ckpt", str(mil_ckpt),
        "--ibot_weights", str(args.ibot_weights),
        "--out_dir", str(out_dir),
        "--mil_arch", "abmil",
        "--out_dim", "3,3",
        "--max_tiles", "64",
        "--num_mispreds", "12",
        "--device", args.device,
    ]

    if args.encoder_ckpt is not None:
        cmd.extend(["--encoder_ckpt", str(args.encoder_ckpt)])

    print(f"[INFO] Prepared {n_images} example ROI images")
    print("[INFO] Running command:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[INFO] Done. Results written to: {out_dir}")


if __name__ == "__main__":
    main()
