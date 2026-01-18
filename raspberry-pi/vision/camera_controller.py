"""
Camera Controller - Main coordinator for dual-mode vision system
Switches between FOLLOW mode (person tracking) and SCAN mode (object detection)
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .aruco_tracker import ArucoTracker, ArucoDetection
from .object_detector import ObjectDetector, ObjectDetection
from .config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class CameraMode(Enum):
    """Camera operating modes"""
    FOLLOW = "follow"  # Track person with colored marker
    SCAN = "scan"      # Detect and identify grocery items


@dataclass
class VisionResult:
    """Unified result from vision system"""
    mode: CameraMode
    found: bool
    label: str
    confidence: float
    center: Optional[Tuple[int, int]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    distance: float = 0.0
    tracking_offset: float = 0.0  # -1 (left) to 1 (right) for robot steering
    raw_detection: Union[ArucoDetection, ObjectDetection, None] = None


class CameraController:
    """
    Main vision system controller
    Manages camera, person tracking, and object detection
    """

    def __init__(self, camera_id: int = 0, use_yolo: bool = True):
        """
        Initialize camera controller

        Args:
            camera_id: Camera device ID (0 for laptop webcam, 0 for Pi)
            use_yolo: Use YOLO for object detection (fallback to color if unavailable)
        """
        self.camera_id = camera_id
        self.mode = CameraMode.SCAN  # Default mode

        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        # Initialize vision modules
        self.aruco_tracker = ArucoTracker()
        self.object_detector = ObjectDetector(use_yolo=use_yolo)

        print(f"✓ CameraController initialized (Camera ID: {camera_id}, Mode: {self.mode.value})")

    def set_mode(self, mode: CameraMode):
        """Switch between FOLLOW and SCAN modes"""
        self.mode = mode
        print(f"✓ Mode changed to: {mode.value.upper()}")

    def calibrate_person_marker(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Calibrate person tracker to marker in center of frame

        Args:
            frame: Optional frame to use, if None will capture new frame

        Returns:
            True if calibration successful
        """
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return False

        return self.aruco_tracker.calibrate(frame)

    def get_follow_distance_m(self, frame: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Get estimated distance (meters) to the ArUco marker.
        Returns None if no marker is detected or calibration is missing.
        """
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return None

        detection = self.aruco_tracker.detect(frame)
        if not detection.found:
            return None

        return detection.distance

    def process_frame(self) -> Tuple[Optional[np.ndarray], VisionResult]:
        """
        Capture and process one frame based on current mode

        Returns:
            (annotated_frame, vision_result)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, VisionResult(
                mode=self.mode,
                found=False,
                label="Camera error",
                confidence=0.0
            )

        # Process based on mode
        if self.mode == CameraMode.FOLLOW:
            result = self._process_follow_mode(frame)
        else:  # SCAN mode
            result = self._process_scan_mode(frame)

        # Annotate frame
        annotated = self._annotate_frame(frame, result)

        return annotated, result

    def _process_follow_mode(self, frame: np.ndarray) -> VisionResult:
        """Process frame in FOLLOW mode"""
        detection = self.aruco_tracker.detect(frame)

        if detection.found:
            # Calculate steering offset
            offset = 0.0
            if detection.center:
                offset = (detection.center[0] - CAMERA_WIDTH / 2) / (CAMERA_WIDTH / 2)

            distance_m = detection.distance if detection.distance is not None else 0.0

            return VisionResult(
                mode=CameraMode.FOLLOW,
                found=True,
                label="Aruco Marker",
                confidence=detection.confidence,
                center=detection.center,
                bbox=detection.bbox,
                distance=distance_m,
                tracking_offset=offset,
                raw_detection=detection
            )
        else:
            return VisionResult(
                mode=CameraMode.FOLLOW,
                found=False,
                label="No target",
                confidence=0.0,
                raw_detection=detection
            )

    def _process_scan_mode(self, frame: np.ndarray) -> VisionResult:
        """Process frame in SCAN mode"""
        detection = self.object_detector.get_best_detection(frame)

        if detection and detection.found:
            # Calculate offset for centering on object
            offset = 0.0
            if detection.center:
                offset = (detection.center[0] - CAMERA_WIDTH / 2) / (CAMERA_WIDTH / 2)

            return VisionResult(
                mode=CameraMode.SCAN,
                found=True,
                label=detection.label,
                confidence=detection.confidence,
                center=detection.center,
                bbox=detection.bbox,
                distance=detection.distance,
                tracking_offset=offset,
                raw_detection=detection
            )
        else:
            return VisionResult(
                mode=CameraMode.SCAN,
                found=False,
                label="No objects detected",
                confidence=0.0
            )

    def _annotate_frame(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """Draw annotations on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Mode indicator (top left)
        mode_color = (255, 0, 255) if result.mode == CameraMode.FOLLOW else (0, 255, 0)
        cv2.putText(
            annotated,
            f"MODE: {result.mode.value.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            mode_color,
            2
        )

        # Calibration status (FOLLOW mode only)
        if result.mode == CameraMode.FOLLOW:
            calibrated = self.aruco_tracker.focal_length_px is not None
            calib_text = "CALIBRATED" if calibrated else "PRESS 'C' TO CALIBRATE"
            calib_color = (0, 255, 0) if calibrated else (0, 0, 255)
            cv2.putText(
                annotated,
                calib_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                calib_color,
                1
            )

        # YOLO method indicator (SCAN mode only)
        if result.mode == CameraMode.SCAN and isinstance(result.raw_detection, ObjectDetection):
            method_text = f"Method: {result.raw_detection.method.upper()}"
            cv2.putText(
                annotated,
                method_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Center crosshair
        cv2.line(annotated, (w//2-20, h//2), (w//2+20, h//2), (255, 255, 255), 1)
        cv2.line(annotated, (w//2, h//2-20), (w//2, h//2+20), (255, 255, 255), 1)

        # Detection visualization
        if result.found and result.bbox:
            x, y, bw, bh = result.bbox

            # Bounding box
            box_color = (0, 255, 0) if result.confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), box_color, 2)

            # Center point
            if result.center:
                cv2.circle(annotated, result.center, 8, (0, 0, 255), -1)

            # Label with confidence
            label_text = f"{result.label} ({result.confidence*100:.0f}%)"
            cv2.putText(
                annotated,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # Distance
            dist_text = f"Dist: {result.distance:.1f}m"
            if isinstance(result.raw_detection, ArucoDetection) and result.raw_detection.distance is None:
                dist_text = "Dist: --"
            cv2.putText(
                annotated,
                dist_text,
                (x, y + bh + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Locked marker center (FOLLOW mode only)
            if result.mode == CameraMode.FOLLOW:
                locked_center = self.aruco_tracker.get_locked_center(frame, normalized=True)
                if locked_center is not None:
                    coord_text = f"X: {locked_center[0]:+.2f} Y: {locked_center[1]:+.2f}"
                    cv2.putText(
                        annotated,
                        coord_text,
                        (x, y + bh + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

            # Offset indicator (for robot steering)
            offset_x = int(w/2 + result.tracking_offset * w/2)
            cv2.arrowedLine(
                annotated,
                (w//2, h - 30),
                (offset_x, h - 30),
                (0, 255, 255),
                3,
                tipLength=0.3
            )
        else:
            # No detection status
            cv2.putText(
                annotated,
                result.label,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return annotated

    def release(self):
        """Release camera resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released")

    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap.isOpened()
