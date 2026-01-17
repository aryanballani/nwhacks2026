#!/usr/bin/env python3
"""
Dual-Mode Camera System: Follow Person OR Scan Objects
Works on both laptop (testing) and Raspberry Pi (production)
"""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import time


class CameraMode(Enum):
    FOLLOW = "follow"  # Follow person with colored marker
    SCAN = "scan"      # Scan and identify grocery items


@dataclass
class DetectionResult:
    found: bool
    label: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    center: Optional[Tuple[int, int]] = None
    distance_estimate: float = 0.0


class DualModeCamera:
    """Camera system that switches between person-following and object-scanning"""

    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize camera system

        Args:
            camera_id: Camera device ID (0 for laptop webcam, 0 for Pi camera)
            width: Frame width
            height: Frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.mode = CameraMode.SCAN  # Start in scan mode

        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Person tracking state
        self.target_color_lower = np.array([140, 100, 100])  # Pink/Magenta
        self.target_color_upper = np.array([170, 255, 255])
        self.calibrated = False

        # Object detection - we'll use color-based detection for common items
        self.item_colors = {
            "banana": {"lower": np.array([20, 100, 100]), "upper": np.array([30, 255, 255])},
            "orange": {"lower": np.array([10, 100, 100]), "upper": np.array([20, 255, 255])},
            "apple_red": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
            "apple_green": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
            "tomato": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
        }

        print(f"✓ Camera initialized (Mode: {self.mode.value})")

    def set_mode(self, mode: CameraMode):
        """Switch between FOLLOW and SCAN modes"""
        self.mode = mode
        print(f"✓ Mode switched to: {mode.value.upper()}")

    def calibrate_person_marker(self):
        """Calibrate to the colored marker in center of frame"""
        ret, frame = self.cap.read()
        if not ret:
            return False

        # Get center region
        h, w = frame.shape[:2]
        center_region = frame[h//2-50:h//2+50, w//2-50:w//2+50]

        # Convert to HSV and get average color
        hsv_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv_region, axis=(0, 1))

        # Set bounds with tolerance
        tolerance = np.array([15, 50, 50])
        self.target_color_lower = np.clip(avg_hsv - tolerance, 0, 255).astype(np.uint8)
        self.target_color_upper = np.clip(avg_hsv + tolerance, 0, 255).astype(np.uint8)

        self.calibrated = True
        print(f"✓ Calibrated to HSV: {avg_hsv}")
        return True

    def detect_person(self, frame) -> DetectionResult:
        """Detect person with colored marker (FOLLOW mode)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for target color
        mask = cv2.inRange(hsv, self.target_color_lower, self.target_color_upper)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return DetectionResult(found=False, label="No target", confidence=0.0)

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 500:  # Filter noise
            return DetectionResult(found=False, label="Target too small", confidence=0.0)

        # Get centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return DetectionResult(found=False, label="Invalid contour", confidence=0.0)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Estimate distance based on marker size
        distance = self._estimate_distance_from_area(area)

        return DetectionResult(
            found=True,
            label="Person (marker detected)",
            confidence=min(area / 10000, 1.0),
            bbox=(x, y, w, h),
            center=(cx, cy),
            distance_estimate=distance
        )

    def detect_object(self, frame) -> DetectionResult:
        """Detect grocery items (SCAN mode)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_detection = None
        best_area = 0

        # Check each item color
        for item_name, color_range in self.item_colors.items():
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

            if area > 1000 and area > best_area:  # Minimum size and best so far
                M = cv2.moments(largest_contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest_contour)

                best_area = area
                best_detection = DetectionResult(
                    found=True,
                    label=item_name.replace("_", " ").title(),
                    confidence=min(area / 5000, 1.0),
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                    distance_estimate=self._estimate_distance_from_area(area)
                )

        if best_detection:
            return best_detection

        return DetectionResult(found=False, label="No items detected", confidence=0.0)

    def _estimate_distance_from_area(self, area: float) -> float:
        """Rough distance estimation based on object size"""
        if area < 1000:
            return 3.0  # Far
        elif area < 5000:
            return 2.0
        elif area < 15000:
            return 1.0
        else:
            return 0.5  # Close

    def process_frame(self) -> Tuple[Optional[np.ndarray], DetectionResult]:
        """
        Process one frame based on current mode

        Returns:
            (annotated_frame, detection_result)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, DetectionResult(found=False, label="Camera error", confidence=0.0)

        # Detect based on mode
        if self.mode == CameraMode.FOLLOW:
            result = self.detect_person(frame)
        else:  # SCAN mode
            result = self.detect_object(frame)

        # Annotate frame
        annotated = self._annotate_frame(frame, result)

        return annotated, result

    def _annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detection results on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw mode indicator
        mode_color = (0, 255, 0) if self.mode == CameraMode.SCAN else (255, 0, 255)
        cv2.putText(annotated, f"MODE: {self.mode.value.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

        # Draw center crosshair
        cv2.line(annotated, (w//2-20, h//2), (w//2+20, h//2), (255, 255, 255), 1)
        cv2.line(annotated, (w//2, h//2-20), (w//2, h//2+20), (255, 255, 255), 1)

        if result.found and result.bbox:
            x, y, w, h = result.bbox

            # Draw bounding box
            color = (0, 255, 0) if result.confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Draw center point
            if result.center:
                cv2.circle(annotated, result.center, 8, (0, 0, 255), -1)

            # Draw label and info
            label_text = f"{result.label} ({result.confidence*100:.0f}%)"
            cv2.putText(annotated, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw distance
            dist_text = f"Dist: {result.distance_estimate:.1f}m"
            cv2.putText(annotated, dist_text, (x, y+h+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Show status
            cv2.putText(annotated, result.label, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw calibration status in FOLLOW mode
        if self.mode == CameraMode.FOLLOW:
            calib_text = "CALIBRATED" if self.calibrated else "NEEDS CALIBRATION (Press C)"
            calib_color = (0, 255, 0) if self.calibrated else (0, 0, 255)
            cv2.putText(annotated, calib_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, calib_color, 1)

        return annotated

    def release(self):
        """Release camera resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released")


def main():
    """Test the dual-mode camera system"""
    print("=" * 60)
    print("GROCERY BUDDY - Dual-Mode Camera Test")
    print("=" * 60)
    print("\nControls:")
    print("  M - Toggle mode (FOLLOW ↔ SCAN)")
    print("  C - Calibrate person marker (FOLLOW mode only)")
    print("  Q - Quit")
    print("  SPACE - Take screenshot")
    print("\nMode descriptions:")
    print("  FOLLOW - Track person with colored marker (pink/magenta)")
    print("  SCAN   - Detect grocery items (banana, apple, orange, etc.)")
    print("=" * 60)

    camera = DualModeCamera(camera_id=0)

    try:
        while True:
            frame, result = camera.process_frame()

            if frame is None:
                print("Failed to read frame")
                break

            # Display frame
            cv2.imshow("Grocery Buddy Camera", frame)

            # Print detection info
            if result.found:
                print(f"\r{result.label} | Conf: {result.confidence*100:.0f}% | Dist: {result.distance_estimate:.1f}m", end="")

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('m'):
                # Toggle mode
                new_mode = CameraMode.SCAN if camera.mode == CameraMode.FOLLOW else CameraMode.FOLLOW
                camera.set_mode(new_mode)
            elif key == ord('c'):
                # Calibrate (FOLLOW mode only)
                if camera.mode == CameraMode.FOLLOW:
                    camera.calibrate_person_marker()
                else:
                    print("\nCalibration only works in FOLLOW mode")
            elif key == ord(' '):
                # Screenshot
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n✓ Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        camera.release()
        print("\nTest complete!")


if __name__ == "__main__":
    main()
