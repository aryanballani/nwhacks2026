"""
ArUco marker tracker for FOLLOW mode calibration and distance estimation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import ARUCO_CALIBRATION_DISTANCE_CM, ARUCO_MARKER_LENGTH_CM


@dataclass
class ArucoDetection:
    """Result of ArUco marker detection"""
    found: bool
    center: Optional[Tuple[int, int]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    distance: Optional[float] = None  # meters
    confidence: float = 0.0
    marker_id: Optional[int] = None


class ArucoTracker:
    """Detects ArUco markers and estimates distance after calibration."""

    def __init__(
        self,
        marker_length_cm: float = ARUCO_MARKER_LENGTH_CM,
        calibration_distance_cm: float = ARUCO_CALIBRATION_DISTANCE_CM,
        aruco_dict_id: int = cv2.aruco.DICT_5X5_50,
    ):
        self.marker_length_cm = float(marker_length_cm)
        self.calibration_distance_cm = float(calibration_distance_cm)
        self.focal_length_px: Optional[float] = None

        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        except AttributeError:
            try:
                self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
            except AttributeError:
                self.aruco_dict = getattr(cv2.aruco, "Dictionary", None)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            try:
                self.aruco_params = cv2.aruco.DetectorParameters()
            except Exception:
                self.aruco_params = None

    def _detect_markers(self, gray: np.ndarray):
        """Detect ArUco markers with compatibility across OpenCV versions."""
        try:
            if hasattr(cv2.aruco, "ArucoDetector") and self.aruco_params is not None:
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                res = detector.detectMarkers(gray)
                if isinstance(res, tuple) and len(res) >= 2:
                    corners, ids = res[0], res[1]
                else:
                    return None, None
            else:
                if self.aruco_params is not None:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        except Exception:
            try:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            except Exception:
                return None, None

        return corners, ids

    def calibrate(self, frame: np.ndarray, known_distance_cm: Optional[float] = None) -> bool:
        """
        Calibrate focal length using an ArUco marker visible in the frame.
        Returns True if calibration succeeds.
        """
        if known_distance_cm is None:
            known_distance_cm = self.calibration_distance_cm

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect_markers(gray)
        if ids is None or len(corners) == 0:
            return False

        c = corners[0].reshape((4, 2))
        top_w = np.linalg.norm(c[0] - c[1])
        bot_w = np.linalg.norm(c[2] - c[3])
        px_w = (top_w + bot_w) / 2.0
        if px_w <= 0:
            return False

        self.focal_length_px = (px_w * float(known_distance_cm)) / self.marker_length_cm
        return True

    def estimate_distance_m(self, pixel_width_px: float) -> Optional[float]:
        """Estimate distance (meters) given marker width in pixels."""
        if not self.focal_length_px or pixel_width_px <= 0:
            return None
        distance_cm = (self.marker_length_cm * self.focal_length_px) / float(pixel_width_px)
        return distance_cm / 100.0

    def get_locked_center(self, frame: np.ndarray, normalized: bool = False) -> Optional[Tuple[float, float]]:
        """
        Return the marker center coordinates when calibrated ("locked").
        If normalized=True, returns (-1..1) offsets from frame center.
        """
        if self.focal_length_px is None:
            return None

        detection = self.detect(frame)
        if not detection.found or detection.center is None:
            return None

        cx, cy = detection.center
        if not normalized:
            return float(cx), float(cy)

        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return None

        x_norm = (cx - w / 2) / (w / 2)
        y_norm = (cy - h / 2) / (h / 2)
        return float(x_norm), float(y_norm)

    def detect(self, frame: np.ndarray) -> ArucoDetection:
        """Detect the first ArUco marker in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect_markers(gray)
        if ids is None or len(corners) == 0:
            return ArucoDetection(found=False)

        c = corners[0].reshape((4, 2))
        center = c.mean(axis=0)
        center_pt = (int(center[0]), int(center[1]))

        x_min = int(np.min(c[:, 0]))
        y_min = int(np.min(c[:, 1]))
        x_max = int(np.max(c[:, 0]))
        y_max = int(np.max(c[:, 1]))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        top_w = np.linalg.norm(c[0] - c[1])
        bot_w = np.linalg.norm(c[2] - c[3])
        px_w = (top_w + bot_w) / 2.0

        distance_m = self.estimate_distance_m(px_w)
        marker_id = int(ids[0][0]) if ids is not None else None

        return ArucoDetection(
            found=True,
            center=center_pt,
            bbox=bbox,
            distance=distance_m,
            confidence=1.0,
            marker_id=marker_id,
        )
