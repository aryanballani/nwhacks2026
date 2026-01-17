#!/usr/bin/env python3
"""
Grocery Buddy - Main Entry Point
Autonomous shopping cart robot with person-following and object scanning
"""

import cv2
import time
import sys
from typing import Optional

from vision import CameraController, CameraMode, VisionResult
from motors.motor_controller import MotorController


class RobotController:
    """Main robot controller integrating vision and motor control"""

    def __init__(self, camera_id: int = 0, use_yolo: bool = True):
        """
        Initialize robot controller

        Args:
            camera_id: Camera device ID
            use_yolo: Use YOLO for object detection (fallback to color if unavailable)
        """
        print("=" * 70)
        print("GROCERY BUDDY - Autonomous Shopping Cart Robot")
        print("=" * 70)
        print()

        # Initialize subsystems
        self.camera = CameraController(camera_id=camera_id, use_yolo=use_yolo)
        self.motors = MotorController()

        # Control parameters
        self.target_distance = 1.0  # Target following distance in meters
        self.distance_tolerance = 0.3  # Tolerance for distance control

        # Speed control
        self.base_speed = 50  # Base motor speed (0-100)
        self.max_speed = 80   # Maximum motor speed
        self.turn_gain = 60   # Steering sensitivity (higher = sharper turns)

        # Safety
        self.min_distance = 0.4  # Minimum safe distance (meters)
        self.max_tracking_age = 2.0  # Max time without detection before stopping

        # State
        self.tracking_enabled = False
        self.last_detection_time = 0
        self.emergency_stop = False

        print("✅ RobotController initialized")
        print(f"📷 Camera mode: {self.camera.mode.value.upper()}")
        print(f"🎯 Target distance: {self.target_distance}m")
        print()

    def calculate_motor_speeds(self, result: VisionResult) -> tuple[float, float]:
        """
        Calculate left and right motor speeds based on vision result

        Args:
            result: Vision detection result

        Returns:
            (left_speed, right_speed) tuple (-100 to 100)
        """
        if not result.found or self.emergency_stop:
            return (0.0, 0.0)

        # Distance-based speed control
        distance_error = result.distance - self.target_distance

        # Forward/backward speed
        if abs(distance_error) < self.distance_tolerance:
            # Within tolerance - maintain position
            forward_speed = 0
        elif distance_error > 0:
            # Too far - move forward
            forward_speed = min(self.base_speed * (distance_error / self.target_distance), self.max_speed)
        else:
            # Too close - move backward (or stop if very close)
            if result.distance < self.min_distance:
                forward_speed = 0
                self.motors.stop()
                return (0.0, 0.0)
            else:
                forward_speed = max(-self.base_speed * 0.5, -30)  # Limited reverse speed

        # Steering based on tracking offset (-1 to 1)
        # Negative offset = target on left, need to turn left
        # Positive offset = target on right, need to turn right
        turn_amount = result.tracking_offset * self.turn_gain

        # Differential drive: adjust left/right speeds for steering
        left_speed = forward_speed - turn_amount
        right_speed = forward_speed + turn_amount

        # Clamp to motor limits
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        return (left_speed, right_speed)

    def process_vision_result(self, result: VisionResult):
        """
        Process vision result and update motor control

        Args:
            result: Current vision detection result
        """
        if result.found:
            self.last_detection_time = time.time()

            if self.tracking_enabled:
                # Calculate and apply motor speeds
                left_speed, right_speed = self.calculate_motor_speeds(result)
                self.motors.set_motors(left_speed, right_speed)
        else:
            # No detection
            time_since_detection = time.time() - self.last_detection_time

            if time_since_detection > self.max_tracking_age:
                # Lost target - stop motors
                self.motors.stop()

    def run_headless(self):
        """
        Run robot in headless mode (no display)
        For Raspberry Pi deployment
        """
        print("🤖 Running in HEADLESS mode (no display)")
        print("Press Ctrl+C to stop\n")

        try:
            frame_count = 0
            while True:
                # Process frame
                frame, result = self.camera.process_frame()

                if frame is None:
                    print("❌ Failed to read frame")
                    break

                # Process vision result
                self.process_vision_result(result)

                # Print status every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    status = "✓" if result.found else "⚪"
                    print(
                        f"{status} {result.label:20s} | "
                        f"Dist: {result.distance:.1f}m | "
                        f"Offset: {result.tracking_offset:+.2f} | "
                        f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}"
                    )

                time.sleep(0.01)  # Small delay for CPU

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            self.shutdown()

    def run_interactive(self):
        """
        Run robot with visual feedback
        For laptop testing
        """
        print("🎮 Running in INTERACTIVE mode (with display)")
        print("\n🎮 CONTROLS:")
        print("  M     - Toggle mode (FOLLOW ↔ SCAN)")
        print("  C     - Calibrate person marker (FOLLOW mode only)")
        print("  T     - Toggle tracking ON/OFF")
        print("  E     - Emergency stop (toggle)")
        print("  Q/ESC - Quit")
        print("  SPACE - Take screenshot")
        print("\n⚠️  Tracking is OFF by default. Press 'T' to enable.\n")

        try:
            frame_count = 0
            fps_time = time.time()

            while True:
                # Process frame
                frame, result = self.camera.process_frame()

                if frame is None:
                    print("❌ Failed to read frame")
                    break

                # Process vision result (only if tracking enabled)
                self.process_vision_result(result)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.time()
                    fps = 30 / (now - fps_time) if fps_time else 0
                    fps_time = now

                    # Print detection info
                    if result.found:
                        print(f"\r{'  ' * 15}", end='')
                        print(
                            f"\r✓ {result.label} | "
                            f"Conf: {result.confidence*100:.0f}% | "
                            f"Dist: {result.distance:.1f}m | "
                            f"Offset: {result.tracking_offset:+.2f} | "
                            f"FPS: {fps:.1f}",
                            end=''
                        )
                    else:
                        print(f"\r{'  ' * 15}", end='')
                        print(f"\r⚪ {result.label} | FPS: {fps:.1f}", end='')

                # Annotate frame with tracking status
                annotated = frame.copy()
                h, w = frame.shape[:2]

                # Tracking status (top right)
                tracking_text = "TRACKING: ON" if self.tracking_enabled else "TRACKING: OFF"
                tracking_color = (0, 255, 0) if self.tracking_enabled else (0, 0, 255)
                cv2.putText(
                    annotated,
                    tracking_text,
                    (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    tracking_color,
                    2
                )

                # Emergency stop indicator
                if self.emergency_stop:
                    cv2.putText(
                        annotated,
                        "EMERGENCY STOP",
                        (w//2 - 150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

                # Display frame
                cv2.imshow("Grocery Buddy Robot Controller", annotated)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break

                elif key == ord('m'):
                    # Toggle mode
                    new_mode = CameraMode.SCAN if self.camera.mode == CameraMode.FOLLOW else CameraMode.FOLLOW
                    self.camera.set_mode(new_mode)
                    print(f"\n\n🔄 Mode switched to: {new_mode.value.upper()}\n")

                elif key == ord('c'):
                    # Calibrate (FOLLOW mode only)
                    if self.camera.mode == CameraMode.FOLLOW:
                        print("\n\n📸 Calibrating... Hold marker in center of frame...")
                        success = self.camera.calibrate_person_marker(frame)
                        if success:
                            print("✅ Calibration successful!\n")
                        else:
                            print("❌ Calibration failed\n")
                    else:
                        print("\n\n⚠️  Calibration only works in FOLLOW mode (press 'M' to switch)\n")

                elif key == ord('t'):
                    # Toggle tracking
                    self.tracking_enabled = not self.tracking_enabled
                    if not self.tracking_enabled:
                        self.motors.stop()
                    status = "ON" if self.tracking_enabled else "OFF"
                    print(f"\n\n🎯 Tracking: {status}\n")

                elif key == ord('e'):
                    # Emergency stop
                    self.emergency_stop = not self.emergency_stop
                    if self.emergency_stop:
                        self.motors.stop()
                    status = "ACTIVATED" if self.emergency_stop else "DEACTIVATED"
                    print(f"\n\n🚨 Emergency Stop: {status}\n")

                elif key == ord(' '):
                    # Screenshot
                    timestamp = int(time.time())
                    filename = f"robot_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"\n\n📸 Screenshot saved: {filename}\n")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all subsystems"""
        print("\n\n🛑 Shutting down...")
        self.motors.stop()
        self.motors.cleanup()
        self.camera.release()
        print("✅ Shutdown complete")
        print("\nGoodbye! 👋\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Grocery Buddy Robot Controller")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (for Raspberry Pi)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO, use color detection only"
    )

    args = parser.parse_args()

    try:
        # Initialize robot controller
        robot = RobotController(
            camera_id=args.camera,
            use_yolo=not args.no_yolo
        )

        # Run in appropriate mode
        if args.headless:
            robot.run_headless()
        else:
            robot.run_interactive()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
