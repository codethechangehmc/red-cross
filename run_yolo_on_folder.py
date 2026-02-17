#!/usr/bin/env python3
"""Run YOLOv8 inference on a folder of images and save annotated outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO on a folder of images.")
    parser.add_argument(
        "--model",
        default="yolo_best.pt",
        help="Path to YOLO model weights (default: yolo_best.pt)",
    )
    parser.add_argument(
        "--source",
        default="equirectangular",
        help="Folder with images to run inference on",
    )
    parser.add_argument(
        "--output",
        default="yolo_predictions_equirectangular",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (e.g., 0, cpu). Default: auto",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    model_path = Path(args.model).expanduser().resolve()
    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source folder not found: {source_dir}")
        return 1

    model = YOLO(str(model_path))

    model.predict(
        source=str(source_dir),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="results",
        exist_ok=True,
        verbose=False,
    )

    print(f"Saved predictions to: {output_dir / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
