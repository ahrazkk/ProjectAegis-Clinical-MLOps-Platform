"""
Create a cleaned YOLO dataset by filtering noisy pill images.

This script removes likely non-pill images using filename heuristics,
then copies remaining images/labels into a new dataset folder.

Expected input:
  <source_root>/images/train, <source_root>/images/val
  <source_root>/labels/train, <source_root>/labels/val
  <source_root>/data.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

NOISY_NAME_TOKENS = {
    'label',
    'directions',
    'instruction',
    'insert',
    'tablet',
    'tablets',
    'caplet',
    'caplets',
    'capsule',
    'capsules',
    'box',
    'carton',
    'blister',
    'foil',
    'rx',
    'w80',
    'w24',
    'tabs',
    'chew',
    'syrup',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Filter noisy images from YOLO pill dataset')
    parser.add_argument(
        '--source-root',
        default=r'c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\web\data\pill_yolo',
        help='Input YOLO dataset root',
    )
    parser.add_argument(
        '--output-root',
        default=r'c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\web\data\pill_yolo_clean',
        help='Output YOLO dataset root',
    )
    return parser.parse_args()


def is_likely_noisy(filename: str) -> bool:
    lower = filename.lower()
    return any(token in lower for token in NOISY_NAME_TOKENS)


def ensure_dirs(root: Path) -> None:
    for split in ('train', 'val'):
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    if not source_root.exists():
        print(f'Source dataset not found: {source_root}')
        return 1

    ensure_dirs(output_root)

    kept = {'train': 0, 'val': 0}
    dropped = {'train': 0, 'val': 0}

    for split in ('train', 'val'):
        src_img_dir = source_root / 'images' / split
        src_lbl_dir = source_root / 'labels' / split
        dst_img_dir = output_root / 'images' / split
        dst_lbl_dir = output_root / 'labels' / split

        for img_path in src_img_dir.glob('*'):
            if not img_path.is_file():
                continue

            label_path = src_lbl_dir / f'{img_path.stem}.txt'
            if not label_path.exists():
                dropped[split] += 1
                continue

            if is_likely_noisy(img_path.name):
                dropped[split] += 1
                continue

            shutil.copy2(img_path, dst_img_dir / img_path.name)
            shutil.copy2(label_path, dst_lbl_dir / label_path.name)
            kept[split] += 1

    src_yaml = source_root / 'data.yaml'
    dst_yaml = output_root / 'data.yaml'
    if src_yaml.exists():
        content = src_yaml.read_text(encoding='utf-8')
        # Rewrite dataset root path in YAML.
        content = content.replace(source_root.as_posix(), output_root.as_posix())
        content = content.replace(str(source_root), str(output_root))
        dst_yaml.write_text(content, encoding='utf-8')

    print('Cleaned YOLO dataset created.')
    print(f'Train kept: {kept["train"]}, dropped: {dropped["train"]}')
    print(f'Val kept: {kept["val"]}, dropped: {dropped["val"]}')
    print(f'Data YAML: {dst_yaml}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
