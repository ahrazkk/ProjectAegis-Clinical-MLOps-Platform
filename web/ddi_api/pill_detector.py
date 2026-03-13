"""
YOLO-based pill detector adapter.

This module is intentionally optional:
- If ultralytics is not installed, calls return empty detections.
- If model file is missing, calls return empty detections.

It allows the API layer to integrate a real detector without breaking existing flows.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Dict, List, Tuple


class YoloPillDetector:
    def __init__(self) -> None:
        self._model = None
        self._load_error = None
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        def resolve_path(p: str) -> str:
            return p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p))

        env_path = os.environ.get('PILL_DETECTOR_MODEL_PATH')
        candidates = []
        if env_path:
            candidates.append(resolve_path(env_path))
        candidates.extend([
            resolve_path(os.path.join('models', 'runs', 'pill', 'tuned-yolo-detector-v3-clean', 'weights', 'best.pt')),
            resolve_path(os.path.join('models', 'runs', 'pill', 'tuned-yolo-detector-v2', 'weights', 'best.pt')),
            resolve_path(os.path.join('models', 'runs', 'pill', 'bootstrap-yolo-detector', 'weights', 'best.pt')),
            resolve_path(os.path.join('models', 'pill_detector', 'best.pt')),
        ])

        existing = [p for p in candidates if os.path.exists(p)]
        self.model_path = existing[0] if existing else candidates[0]

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        if not os.path.exists(self.model_path):
            self._load_error = f'Model not found: {self.model_path}'
            return False

        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self.model_path)
            return True
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._load_error = str(exc)
            return False

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def detect_from_upload(self, image_bytes: bytes, max_det: int = 3) -> List[Dict]:
        """Run detection on uploaded image bytes and return normalized boxes."""
        if not image_bytes or not self._ensure_loaded():
            return []

        suffix = '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        try:
            results = self._model.predict(temp_path, verbose=False, max_det=max_det)
            if not results:
                return []

            result = results[0]
            boxes = getattr(result, 'boxes', None)
            if boxes is None or boxes.xyxy is None:
                return []

            names = getattr(result, 'names', {}) or {}
            detections: List[Dict] = []
            img_h, img_w = self._get_image_size(result)

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                cls_idx = int(boxes.cls[i].item()) if boxes.cls is not None else -1
                cls_name = names.get(cls_idx, f'class_{cls_idx}')

                x1, y1, x2, y2 = xyxy
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                detections.append({
                    'class_index': cls_idx,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': {
                        'x': x1,
                        'y': y1,
                        'width': w,
                        'height': h,
                        'x_norm': x1 / img_w if img_w else 0,
                        'y_norm': y1 / img_h if img_h else 0,
                        'width_norm': w / img_w if img_w else 0,
                        'height_norm': h / img_h if img_h else 0,
                    },
                })

            detections.sort(key=lambda d: d['confidence'], reverse=True)
            return detections
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _get_image_size(self, result) -> Tuple[float, float]:
        orig_shape = getattr(result, 'orig_shape', None)
        if isinstance(orig_shape, (tuple, list)) and len(orig_shape) == 2:
            return float(orig_shape[0]), float(orig_shape[1])
        return 1.0, 1.0


pill_detector = YoloPillDetector()
