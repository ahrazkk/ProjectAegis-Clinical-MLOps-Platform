"""
Train a YOLO pill detector/segmenter on a YOLO-format dataset.

Usage:
  python train_pill_yolo.py --data ../data/pill_yolo/data.yaml --task detect
  python train_pill_yolo.py --data ../data/pill_yolo/data.yaml --task segment
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train YOLO for pill localization and identity')
    parser.add_argument('--data', required=True, help='Path to YOLO dataset YAML')
    parser.add_argument('--task', choices=['detect', 'segment'], default='detect')
    parser.add_argument('--model', default='yolov8n.pt', help='Base pretrained model')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', default='runs/pill')
    parser.add_argument('--name', default='pill-detector')
    parser.add_argument('--device', default='cpu', help="cpu or cuda index like '0'")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.data):
        print(f'Dataset YAML not found: {args.data}')
        return 1

    try:
        YOLO = importlib.import_module('ultralytics').YOLO
    except Exception as exc:
        print('Ultralytics is required. Install with: pip install ultralytics')
        print(f'Import error: {exc}')
        return 1

    model = YOLO(args.model)
    model.train(
        data=args.data,
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    print('Training complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
