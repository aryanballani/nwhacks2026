"""
Object Detector - YOLO-based grocery item detection with color fallback
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from .config import (
    YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD,
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
    """Detects grocery items using YOLO (primary) or color detection (fallback)"""

    def __init__(self, use_yolo: bool = True):
        """
        Initialize object detector

        Args:
            use_yolo: Try to use YOLO if available, fallback to color detection
        """
        self.yolo_available = False
        self.model = None

        if use_yolo:
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

        if not self.yolo_available:
            print("✓ ObjectDetector initialized (color-based mode)")

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
        # Try YOLO first
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
