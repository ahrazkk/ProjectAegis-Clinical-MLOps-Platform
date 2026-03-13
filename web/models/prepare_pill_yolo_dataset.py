"""
Prepare a YOLO detection dataset from class-folder pill images.

Input layout expected:
  <source_root>/train/<class_name>/*.jpg
  <source_root>/val/<class_name>/*.jpg

Output layout:
  <output_root>/images/train/*.jpg
  <output_root>/images/val/*.jpg
  <output_root>/labels/train/*.txt
  <output_root>/labels/val/*.txt
  <output_root>/data.yaml

Bootstrapping note:
- This script creates a centered weak bounding box for each image.
- It is a practical starting point for detector pretraining before manual box cleanup.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert class folders to YOLO detect dataset')
    parser.add_argument(
        '--source-root',
        default=r'c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\pill_eval_data',
        help='Source folder containing train/ and val/ class directories',
    )
    parser.add_argument(
        '--output-root',
        default=r'c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\web\data\pill_yolo',
        help='Output YOLO dataset root',
    )
    parser.add_argument(
        '--bbox-scale',
        type=float,
        default=0.82,
        help='Relative width/height for centered weak box (0.0-1.0)',
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in name.strip().lower())


def iter_images(class_dir: Path):
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def write_label_file(label_path: Path, class_idx: int, bbox_scale: float) -> None:
    # YOLO format: class cx cy w h (normalized)
    cx = 0.5
    cy = 0.5
    w = max(0.05, min(1.0, bbox_scale))
    h = max(0.05, min(1.0, bbox_scale))
    label_path.write_text(f'{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n', encoding='utf-8')


def ensure_dirs(output_root: Path) -> None:
    for split in ('train', 'val'):
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    bbox_scale = args.bbox_scale

    if not source_root.exists():
        print(f'Source root not found: {source_root}')
        return 1

    train_root = source_root / 'train'
    val_root = source_root / 'val'
    if not train_root.exists() or not val_root.exists():
        print(f'Expected train/ and val/ under {source_root}')
        return 1

    ensure_dirs(output_root)

    class_names = sorted(
        [d.name for d in train_root.iterdir() if d.is_dir() and any(iter_images(d))]
    )

    if not class_names:
        print('No class folders with images found in source train split.')
        return 1

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    copied = {'train': 0, 'val': 0}
    for split, split_root in (('train', train_root), ('val', val_root)):
        for class_name in class_names:
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue

            class_idx = class_to_idx[class_name]
            class_key = sanitize_name(class_name)

            for img in iter_images(class_dir):
                stem = sanitize_name(img.stem)
                dest_name = f'{class_key}__{stem}{img.suffix.lower()}'

                img_dest = output_root / 'images' / split / dest_name
                lbl_dest = output_root / 'labels' / split / f'{Path(dest_name).stem}.txt'

                shutil.copy2(img, img_dest)
                write_label_file(lbl_dest, class_idx, bbox_scale)
                copied[split] += 1

    data_yaml = output_root / 'data.yaml'
    names_lines = '\n'.join(f'  {i}: {sanitize_name(name)}' for i, name in enumerate(class_names))
    data_yaml.write_text(
        '\n'.join([
            f'path: {output_root.as_posix()}',
            'train: images/train',
            'val: images/val',
            'names:',
            names_lines,
            '',
        ]),
        encoding='utf-8',
    )

    print('YOLO dataset prepared successfully.')
    print(f'Classes: {len(class_names)}')
    print(f"Train images: {copied['train']}")
    print(f"Val images: {copied['val']}")
    print(f'Data YAML: {data_yaml}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
