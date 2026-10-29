"""
VisionForge AI - Object Detection Engine

Responsibilities:
- Load a pretrained YOLO model.
- Run inference on an image.
- Extract object labels, confidence scores, and bounding boxes.
- Return annotated images and structured detection results.
"""

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
from ultralytics import YOLO


class DetectionEngine:
    """Reusable engine for YOLO-based object detection."""

    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        self.model_path = model_path

        # Load the model once and reuse it for subsequent predictions.
        self.model = YOLO(model_path)

    def detect_image(
        self,
        image_path: str,
        confidence: float = 0.25,
    ) -> dict[str, Any]:
        """Detect objects in an image and return structured results."""

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0.")

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(f"Could not read image: {path}")

        start_time = perf_counter()

        predictions = self.model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )

        elapsed_seconds = perf_counter() - start_time

        prediction = predictions[0]
        detections = []

        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())

                x1, y1, x2, y2 = (
                    int(value)
                    for value in box.xyxy[0].tolist()
                )

                names = self.model.names
                if isinstance(names, dict):
                    label = str(names.get(class_id, f"class_{class_id}"))
                elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                    label = str(names[class_id])
                else:
                    label = f"class_{class_id}"

                detections.append(
                    {
                        "class_id": class_id,
                        "label": label,
                        "confidence": round(score, 4),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )

        annotated_image = prediction.plot()

        return {
            "image_path": str(path),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "detections": detections,
            "object_count": len(detections),
            "inference_time_ms": round(elapsed_seconds * 1000, 2),
            "annotated_image": annotated_image,
        }