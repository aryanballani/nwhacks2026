"""
Object Detector - ONNX/YOLO-based grocery item detection with color fallback
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from .config import (
    OBJECT_DETECTION_MODEL,
    YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD,
    ONNX_CONFIDENCE_THRESHOLD, ONNX_IOU_THRESHOLD, ONNX_MODELS,
    GROCERY_CLASSES, GROCERY_ITEM_COLORS, MIN_OBJECT_AREA,
    estimate_distance
)


@dataclass
class ObjectDetection:
    """Result of object detection"""
    found: bool
    label: str = ""
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    center: Optional[Tuple[int, int]] = None
    distance: float = 0.0
    method: str = "none"  # "yolo" or "color"


class ObjectDetector:
    """Detects grocery items using ONNX/YOLO (primary) or color detection (fallback)"""

    def __init__(self, use_yolo: bool = True):
        """
        Initialize object detector

        Args:
            use_yolo: Try to use ONNX/YOLO if available, fallback to color detection
        """
        self.onnx_available = False
        self.yolo_available = False
        self.ort_session = None
        self.ort_input_name = None
        self.ort_output_names = None
        self.onnx_input_size = (416, 416)  # (h, w) default if not specified
        self.model = None
        self.model_choice = OBJECT_DETECTION_MODEL
        self._onnx_class_names = None

        if use_yolo:
            if self.model_choice in ONNX_MODELS:
                onnx_cfg = ONNX_MODELS[self.model_choice]
                self.onnx_input_size = (onnx_cfg["input_size"], onnx_cfg["input_size"])
                class_names = onnx_cfg.get("class_names", [])
                self._onnx_class_names = class_names if class_names else None
                print(
                    f"ℹ ONNX selected: {self.model_choice} | "
                    f"path={onnx_cfg['path']} | "
                    f"input={self.onnx_input_size[0]} | "
                    f"classes={len(class_names)}"
                )
                try:
                    self._load_onnx(onnx_cfg["path"])
                except ImportError:
                    print("⚠ onnxruntime not installed, skipping ONNX load")
                    pass
                except Exception as e:
                    print(f"⚠ ONNX loading failed: {e}")
                    print("  Using YOLO/color detection fallback")
            else:
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(YOLO_MODEL)
                    self.yolo_available = True
                    print(f"✓ YOLO model loaded: {YOLO_MODEL}")
                except ImportError:
                    print("⚠ ultralytics not installed, using color detection fallback")
                    print("  Install: pip install ultralytics")
                except Exception as e:
                    print(f"⚠ YOLO loading failed: {e}")
                    print("  Using color detection fallback")

        if use_yolo and not self.onnx_available and not self.yolo_available:
            try:
                from ultralytics import YOLO
                self.model = YOLO(YOLO_MODEL)
                self.yolo_available = True
                print(f"✓ YOLO model loaded: {YOLO_MODEL}")
            except ImportError:
                print("⚠ ultralytics not installed, using color detection fallback")
                print("  Install: pip install ultralytics")
            except Exception as e:
                print(f"⚠ YOLO loading failed: {e}")
                print("  Using color detection fallback")

        if not self.onnx_available and not self.yolo_available:
            print("✓ ObjectDetector initialized (color-based mode)")

    def _load_onnx(self, model_rel_path: str) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        model_path = base_dir / model_rel_path
        if not model_path.exists():
            print(f"⚠ ONNX model not found at: {model_path}")
            return

        import onnxruntime as ort

        self.ort_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        inputs = self.ort_session.get_inputs()
        outputs = self.ort_session.get_outputs()
        self.ort_input_name = inputs[0].name
        self.ort_output_names = [o.name for o in outputs]

        shape = inputs[0].shape
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.onnx_input_size = (shape[2], shape[3])

        self.onnx_available = True
        print(f"✓ ONNX model loaded: {model_path}")

    def _letterbox(self, img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = img.shape[:2]  # (h, w)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, r, (dw, dh)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _resolve_label(self, class_id: int) -> Optional[str]:
        if self._onnx_class_names:
            if 0 <= class_id < len(self._onnx_class_names):
                return self._onnx_class_names[class_id]
            return None

        return GROCERY_CLASSES.get(class_id, f"class_{class_id}")

    def _preprocess_onnx(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        img, ratio, pad = self._letterbox(frame, self.onnx_input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, ratio, pad

    def detect_onnx(self, frame: np.ndarray) -> List[ObjectDetection]:
        if not self.onnx_available or self.ort_session is None:
            return []

        input_tensor, ratio, pad = self._preprocess_onnx(frame)
        outputs = self.ort_session.run(self.ort_output_names, {self.ort_input_name: input_tensor})
        output = outputs[0]

        if output.ndim == 3:
            output = output[0]

        if output.shape[0] < output.shape[1] and output.shape[0] < 128:
            output = output.T

        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        keep = confidences >= ONNX_CONFIDENCE_THRESHOLD
        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        if len(boxes) == 0:
            return []

        # xywh -> xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        keep_indices = self._nms(xyxy, confidences, ONNX_IOU_THRESHOLD)

        detections = []
        for idx in keep_indices:
            class_id = int(class_ids[idx])
            label = self._resolve_label(class_id)
            if label is None:
                continue

            x1, y1, x2, y2 = xyxy[idx]
            x1 = (x1 - pad[0]) / ratio
            y1 = (y1 - pad[1]) / ratio
            x2 = (x2 - pad[0]) / ratio
            y2 = (y2 - pad[1]) / ratio

            x1 = max(0, min(int(x1), frame.shape[1] - 1))
            y1 = max(0, min(int(y1), frame.shape[0] - 1))
            x2 = max(0, min(int(x2), frame.shape[1] - 1))
            y2 = max(0, min(int(y2), frame.shape[0] - 1))

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            cx = x1 + w // 2
            cy = y1 + h // 2

            area = w * h
            distance = estimate_distance(area)

            detections.append(ObjectDetection(
                found=True,
                label=label,
                confidence=float(confidences[idx]),
                bbox=(x1, y1, w, h),
                center=(cx, cy),
                distance=distance,
                method="onnx"
            ))

        return detections

    def detect_yolo(self, frame: np.ndarray) -> List[ObjectDetection]:
        """
        Detect objects using YOLO

        Args:
            frame: BGR image from camera

        Returns:
            List of ObjectDetection results
        """
        if not self.yolo_available or self.model is None:
            return []

        # Run inference
        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False
        )

        detections = []

        # Process results
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                # Get class ID
                class_id = int(box.cls[0])

                # Check if it's a grocery item
                if class_id in GROCERY_CLASSES:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                    # Calculate center
                    cx = x + w // 2
                    cy = y + h // 2

                    # Get confidence
                    confidence = float(box.conf[0])

                    # Estimate distance
                    area = w * h
                    distance = estimate_distance(area)

                    detections.append(ObjectDetection(
                        found=True,
                        label=GROCERY_CLASSES[class_id],
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        center=(cx, cy),
                        distance=distance,
                        method="yolo"
                    ))

        return detections

    def detect_color(self, frame: np.ndarray) -> Optional[ObjectDetection]:
        """
        Detect objects using color-based detection (fallback)

        Args:
            frame: BGR image from camera

        Returns:
            Best ObjectDetection result or None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_detection = None
        best_area = 0

        # Check each grocery item color
        for item_name, color_range in GROCERY_ITEM_COLORS.items():
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])

            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # Find largest contour for this item
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > MIN_OBJECT_AREA and area > best_area:
                M = cv2.moments(largest_contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest_contour)

                best_area = area
                best_detection = ObjectDetection(
                    found=True,
                    label=item_name.replace("_", " ").title(),
                    confidence=min(area / 5000, 1.0),
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                    distance=estimate_distance(area),
                    method="color"
                )

        return best_detection

    def detect(self, frame: np.ndarray, max_results: int = 5) -> List[ObjectDetection]:
        """
        Detect objects using best available method

        Args:
            frame: BGR image from camera
            max_results: Maximum number of detections to return

        Returns:
            List of ObjectDetection results
        """
        # Try ONNX first
        if self.onnx_available:
            detections = self.detect_onnx(frame)
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
            return detections[:max_results]

        # Try YOLO next
        if self.yolo_available:
            detections = self.detect_yolo(frame)
            # Sort by confidence and limit results
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
            return detections[:max_results]

        # Fallback to color detection
        detection = self.detect_color(frame)
        return [detection] if detection else []

    def get_best_detection(self, frame: np.ndarray) -> Optional[ObjectDetection]:
        """
        Get single best detection from frame

        Args:
            frame: BGR image from camera

        Returns:
            Best ObjectDetection or None
        """
        detections = self.detect(frame, max_results=1)
        return detections[0] if detections else None
