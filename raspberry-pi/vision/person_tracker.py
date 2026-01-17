"""
Person Tracker - Color-based marker detection for following
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque
from .config import (
    PERSON_MARKER_COLORS, DEFAULT_MARKER,
    MIN_CONTOUR_AREA, estimate_distance
)


@dataclass
class PersonDetection:
    """Result of person detection"""
    found: bool
    center: Optional[Tuple[int, int]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    area: float = 0.0
    distance: float = 0.0
    confidence: float = 0.0


class PersonTracker:
    """Tracks person using colored marker on their back"""

    def __init__(self, marker_color: str = DEFAULT_MARKER):
        """
        Initialize person tracker

        Args:
            marker_color: Color marker to track (pink_magenta, neon_green, neon_orange)
        """
        self.calibrated = False
        self.marker_color = marker_color

        # Load default color range
        if marker_color in PERSON_MARKER_COLORS:
            color_range = PERSON_MARKER_COLORS[marker_color]
            self.hsv_lower = color_range["lower"].copy()
            self.hsv_upper = color_range["upper"].copy()
        else:
            # Default to pink
            color_range = PERSON_MARKER_COLORS[DEFAULT_MARKER]
            self.hsv_lower = color_range["lower"].copy()
            self.hsv_upper = color_range["upper"].copy()

        # Position smoothing
        self.position_history = deque(maxlen=5)

        print(f"✓ PersonTracker initialized (marker: {marker_color})")

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Calibrate to the marker color in center of frame

        Args:
            frame: BGR image from camera

        Returns:
            True if calibration successful
        """
        h, w = frame.shape[:2]

        # Get center region (100x100 pixels)
        center_region = frame[h//2-50:h//2+50, w//2-50:w//2+50]

        if center_region.size == 0:
            return False

        # Convert to HSV and get average color
        hsv_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv_region, axis=(0, 1))

        # Set bounds with tolerance
        tolerance = np.array([15, 50, 50])
        self.hsv_lower = np.clip(avg_hsv - tolerance, 0, 255).astype(np.uint8)
        self.hsv_upper = np.clip(avg_hsv + tolerance, 0, 255).astype(np.uint8)

        self.calibrated = True
        print(f"✓ Calibrated to HSV: {avg_hsv}")
        return True

    def detect(self, frame: np.ndarray) -> PersonDetection:
        """
        Detect person marker in frame

        Args:
            frame: BGR image from camera

        Returns:
            PersonDetection result
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for target color
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return PersonDetection(found=False)

        # Find largest contour (assumed to be person marker)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Filter out noise
        if area < MIN_CONTOUR_AREA:
            return PersonDetection(found=False)

        # Get centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return PersonDetection(found=False)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Smooth position
        self.position_history.append((cx, cy))
        if len(self.position_history) > 0:
            avg_x = sum(p[0] for p in self.position_history) / len(self.position_history)
            avg_y = sum(p[1] for p in self.position_history) / len(self.position_history)
            smooth_center = (int(avg_x), int(avg_y))
        else:
            smooth_center = (cx, cy)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Estimate distance
        distance = estimate_distance(area)

        # Calculate confidence (0-1 based on size)
        confidence = min(area / 10000, 1.0)

        return PersonDetection(
            found=True,
            center=smooth_center,
            bbox=(x, y, w, h),
            area=area,
            distance=distance,
            confidence=confidence
        )

    def get_tracking_offset(self, detection: PersonDetection, frame_width: int) -> float:
        """
        Calculate horizontal offset for robot steering

        Args:
            detection: PersonDetection result
            frame_width: Width of camera frame

        Returns:
            Offset from -1 (left) to 1 (right), 0 is centered
        """
        if not detection.found or not detection.center:
            return 0.0

        cx = detection.center[0]
        offset = (cx - frame_width / 2) / (frame_width / 2)
        return offset
