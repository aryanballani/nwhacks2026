"""
Camera Controller - Main coordinator for dual-mode vision system
Switches between FOLLOW mode (person tracking) and SCAN mode (object detection)
"""

import cv2
import numpy as np
import logging
import time
import threading
from enum import Enum
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from queue import Queue

from .aruco_tracker import ArucoTracker, ArucoDetection
from .object_detector import ObjectDetector, ObjectDetection
from .config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def __init__(self, camera_id: int = 0, use_yolo: bool = True, threaded: bool = True):
        """
        Initialize camera controller

        Args:
            camera_id: Camera device ID (0 for laptop webcam, 0 for Pi)
            use_yolo: Use YOLO for object detection (fallback to color if unavailable)
            threaded: Use threaded camera capture for better performance
        """
        self.camera_id = camera_id
        self.mode = CameraMode.SCAN  # Default mode
        self.threaded = threaded

        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # Enable camera optimizations for Raspberry Pi
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        # Initialize vision modules
        self.aruco_tracker = ArucoTracker()
        self.object_detector = ObjectDetector(use_yolo=use_yolo)

        # Performance optimization - frame skipping
        self._frame_count = 0
        self._skip_frames_scan = 2  # Process every 3rd frame in SCAN mode (reduce CPU load)
        self._skip_frames_follow = 0  # No skipping in FOLLOW mode (ArUco is fast)
        self._last_result = None  # Cache last result for skipped frames

        # Threading setup
        if self.threaded:
            self._frame_queue = Queue(maxsize=2)  # Small queue to avoid lag
            self._result_queue = Queue(maxsize=2)
            self._stop_event = threading.Event()

            # Start capture thread
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            # Start processing thread
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()

            logger.info("✓ Threaded camera capture enabled")

        # Performance monitoring
        self._timing_stats = {
            "capture_ms": [],
            "process_ms": [],
            "annotate_ms": []
        }

        print(f"✓ CameraController initialized (Camera ID: {camera_id}, Mode: {self.mode.value}, Threaded: {threaded})")

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

    def _capture_loop(self):
        """Background thread for continuous camera capture"""
        logger.info("Camera capture thread started")
        while not self._stop_event.is_set():
            t_start = time.time()
            ret, frame = self.cap.read()
            capture_time = (time.time() - t_start) * 1000

            if ret:
                # Drop old frames if queue is full (keep only latest)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except:
                        pass

                self._frame_queue.put((frame, capture_time))
            else:
                logger.warning("Failed to capture frame")
                time.sleep(0.01)

        logger.info("Camera capture thread stopped")

    def _processing_loop(self):
        """Background thread for frame processing"""
        logger.info("Frame processing thread started")
        from queue import Empty
        while not self._stop_event.is_set():
            try:
                frame, capture_time = self._frame_queue.get(timeout=0.5)

                # Process frame
                t_start = time.time()

                # Determine if we should process this frame (optimization)
                skip_interval = self._skip_frames_follow if self.mode == CameraMode.FOLLOW else self._skip_frames_scan
                should_process = (self._frame_count % (skip_interval + 1)) == 0

                self._frame_count += 1

                # Process based on mode (only if not skipping or no cached result)
                if should_process or self._last_result is None:
                    if self.mode == CameraMode.FOLLOW:
                        result = self._process_follow_mode(frame)
                    else:  # SCAN mode
                        result = self._process_scan_mode(frame)
                    self._last_result = result
                else:
                    # Use cached result
                    result = self._last_result

                process_time = (time.time() - t_start) * 1000

                # Drop old results if queue is full
                if self._result_queue.full():
                    try:
                        self._result_queue.get_nowait()
                    except:
                        pass

                self._result_queue.put((frame, result, capture_time, process_time))

                # Log timing every 30 frames
                if self._frame_count % 30 == 0:
                    logger.debug(f"Capture: {capture_time:.1f}ms, Process: {process_time:.1f}ms")

            except Empty:
                # Timeout waiting for frame, this is normal during startup/shutdown
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Error in processing loop: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.01)

        logger.info("Frame processing thread stopped")

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
        if self.threaded:
            # Get result from processing thread
            from queue import Empty
            try:
                frame, result, capture_time, process_time = self._result_queue.get(timeout=0.5)

                # Annotate frame
                t_start = time.time()
                annotated = self._annotate_frame(frame, result)
                annotate_time = (time.time() - t_start) * 1000

                # Log timing stats every 30 frames
                if self._frame_count % 30 == 0:
                    logger.info(
                        f"[{self.mode.value.upper()}] "
                        f"Capture: {capture_time:.1f}ms | "
                        f"Process: {process_time:.1f}ms | "
                        f"Annotate: {annotate_time:.1f}ms | "
                        f"Total: {capture_time + process_time + annotate_time:.1f}ms"
                    )

                return annotated, result

            except Empty:
                # No frame ready yet, return empty result
                return None, VisionResult(
                    mode=self.mode,
                    found=False,
                    label="Waiting for frame...",
                    confidence=0.0
                )
            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                import traceback
                traceback.print_exc()
                return None, VisionResult(
                    mode=self.mode,
                    found=False,
                    label="Processing error",
                    confidence=0.0
                )
        else:
            # Non-threaded mode (original implementation)
            t_capture_start = time.time()
            ret, frame = self.cap.read()
            capture_time = (time.time() - t_capture_start) * 1000

            if not ret:
                return None, VisionResult(
                    mode=self.mode,
                    found=False,
                    label="Camera error",
                    confidence=0.0
                )

            # Determine if we should process this frame (optimization)
            skip_interval = self._skip_frames_follow if self.mode == CameraMode.FOLLOW else self._skip_frames_scan
            should_process = (self._frame_count % (skip_interval + 1)) == 0

            self._frame_count += 1

            # Process based on mode
            t_process_start = time.time()
            if should_process or self._last_result is None:
                if self.mode == CameraMode.FOLLOW:
                    result = self._process_follow_mode(frame)
                else:  # SCAN mode
                    result = self._process_scan_mode(frame)
                self._last_result = result
            else:
                # Use cached result but update mode if changed
                result = self._last_result
                if result.mode != self.mode:
                    # Mode changed, force reprocess
                    if self.mode == CameraMode.FOLLOW:
                        result = self._process_follow_mode(frame)
                    else:
                        result = self._process_scan_mode(frame)
                    self._last_result = result

            process_time = (time.time() - t_process_start) * 1000

            # Annotate the current frame with latest result
            t_annotate_start = time.time()
            annotated = self._annotate_frame(frame, result)
            annotate_time = (time.time() - t_annotate_start) * 1000

            # Log timing stats every 30 frames
            if self._frame_count % 30 == 0:
                logger.info(
                    f"[{self.mode.value.upper()}] "
                    f"Capture: {capture_time:.1f}ms | "
                    f"Process: {process_time:.1f}ms | "
                    f"Annotate: {annotate_time:.1f}ms | "
                    f"Total: {capture_time + process_time + annotate_time:.1f}ms"
                )

            return annotated, result

    def _process_follow_mode(self, frame: np.ndarray) -> VisionResult:
        """Process frame in FOLLOW mode - ArUco marker tracking"""
        detection = self.aruco_tracker.detect(frame)

        if detection.found:
            # Calculate steering offset
            offset = 0.0
            if detection.center:
                offset = (detection.center[0] - CAMERA_WIDTH / 2) / (CAMERA_WIDTH / 2)

            # Distance estimation (will be 0.0 if not calibrated)
            distance_m = detection.distance if detection.distance is not None else 0.0

            # Determine label based on calibration status
            label = f"ArUco Marker #{detection.marker_id}" if detection.marker_id is not None else "ArUco Marker"
            is_calibrated = self.aruco_tracker.focal_length_px is not None

            if not is_calibrated:
                label += " (Uncalibrated - Press 'C')"
            else:
                label += f" (Calibrated)"

            # Log detection details for debugging
            if self._frame_count % 30 == 0:
                logger.info(
                    f"ArUco Detection: ID={detection.marker_id}, "
                    f"Distance={distance_m:.2f}m, Offset={offset:+.3f}, "
                    f"Calibrated={is_calibrated}, Focal={self.aruco_tracker.focal_length_px}"
                )

            return VisionResult(
                mode=CameraMode.FOLLOW,
                found=True,
                label=label,
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
                label="No ArUco marker detected",
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

            # Bounding box - GREEN for good detection, ORANGE for low confidence
            box_color = (0, 255, 0) if result.confidence > 0.5 else (0, 165, 255)
            # Draw thicker box for better visibility
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), box_color, 3)

            # Center point - RED DOT (very visible)
            if result.center:
                # Draw outer circle for better visibility
                cv2.circle(annotated, result.center, 10, (0, 0, 0), 2)  # Black outline
                cv2.circle(annotated, result.center, 8, (0, 0, 255), -1)  # Red filled

            # Label with confidence
            label_text = f"{result.label} ({result.confidence*100:.0f}%)"
            # Add background to text for better visibility
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - text_height - 15), (x + text_width, y - 5), (0, 0, 0), -1)
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
        # Stop threads if threaded mode is enabled
        if self.threaded:
            logger.info("Stopping camera threads...")
            self._stop_event.set()

            # Wait for threads to finish (with timeout)
            if self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            if self._processing_thread.is_alive():
                self._processing_thread.join(timeout=2.0)

            logger.info("Camera threads stopped")

        self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released")

    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap.isOpened()
