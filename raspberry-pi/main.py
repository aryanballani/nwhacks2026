#!/usr/bin/env python3
"""
Grocery Buddy - Main Entry Point
Autonomous shopping cart robot with person-following and object scanning
"""

import cv2
import time
import sys
import logging
from typing import Optional

from vision import CameraController, CameraMode, VisionResult
from motors.motor_controller import MotorController

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info("Initializing camera controller...")
        self.camera = CameraController(camera_id=camera_id, use_yolo=use_yolo)

        logger.info("Initializing motor controller...")
        try:
            self.motors = MotorController()
            logger.info("✅ Motors initialized successfully")
        except Exception as e:
            logger.error(f"❌ Motor initialization failed: {e}")
            logger.warning("⚠️  Running in CAMERA-ONLY mode (no motors)")
            self.motors = None

        # Control parameters
        self.target_distance = 1.0  # Target following distance in meters
        self.distance_tolerance = 0.3  # Tolerance for distance control

        # Speed control
        self.base_speed = 50  # Base motor speed (0-100)
        self.max_speed = 80   # Maximum motor speed
        self.turn_gain = 80   # Steering sensitivity (higher = sharper turns)

        # Safety
        self.min_distance = 0.4  # Minimum safe distance (meters)
        self.max_tracking_age = 1.0  # Max time without detection before stopping

        # State
        self.tracking_enabled = False
        self.last_detection_time = 0
        self.emergency_stop = False

        print("✅ RobotController initialized")
        print(f"📷 Camera mode: {self.camera.mode.value.upper()}")
        print(f"🎯 Target distance: {self.target_distance}m")
        print(f"🎮 Motors: {'ENABLED' if self.motors else 'DISABLED'}")
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

        # Check if calibrated
        is_calibrated = self.camera.aruco_tracker.focal_length_px is not None

        # Log every 30 frames to avoid spam
        frame_count = getattr(self, '_motor_log_count', 0)
        should_log = frame_count % 30 == 0
        self._motor_log_count = frame_count + 1

        # If NOT calibrated, just track by rotation (no forward/backward)
        if not is_calibrated:
            if should_log:
                logger.warning("⚠️  NOT CALIBRATED - Tracking by rotation only")
                logger.info(f"Offset={result.tracking_offset:+.3f}")

            if result.found:
                # Just rotate to center the marker
                turn_amount = result.tracking_offset * self.turn_gain * 3  # Gentler turns when not calibrated

                left_speed = abs(turn_amount)
                right_speed = abs(turn_amount)

                # Clamp to motor limits
                left_speed = max(-100, min(100, left_speed))  # Lower max speed when not calibrated
                right_speed = max(-100, min(100, right_speed))

                if result.tracking_offset < 0: # left
                    print(f"turning left: {left_speed}", flush=True)
                    self.motors.set_motors(left_speed, right_speed)
                    time.sleep(0.5)
                else:  # right
                    self.motors.set_motors(left_speed-5, right_speed)
                    time.sleep(0.5)
            
                if should_log:
                    direction = "LEFT ⬅️ " if result.tracking_offset < 0 else "RIGHT ➡️"
                    logger.info(f"🔄 Turning {direction} to center marker")
                    logger.info(f"🎮 Motor Speeds: L={left_speed:+.0f}, R={right_speed:+.0f}")
                    print("mottrorrrrrrrr", flush=True)
                    print("motor should move formwr52582828282", flush=True)


                return (left_speed*3, right_speed*3)
            else:
                self.motors.stop()

        # CALIBRATED - Use distance-based control
        distance_error = result.distance - self.target_distance

        if should_log:
            logger.info(
                f"Motor Control: Distance={result.distance:.2f}m "
                f"(target={self.target_distance:.2f}m, error={distance_error:+.2f}m), "
                f"Offset={result.tracking_offset:+.3f}"
            )

        # Forward/backward speed based on distance
        if abs(distance_error) < self.distance_tolerance:
            # Within tolerance - maintain position
            forward_speed = 0
            if should_log:
                logger.info("✓ Distance OK - Maintaining position")
        elif distance_error > 0:
            # Too far - move forward
            print(f"distance error: {distance_error}",  flush=True)
            print(f"target distance: {self.target_distance}", flush=True)
            forward_speed = min(self.base_speed * (distance_error / self.target_distance), self.max_speed)
            print(f"forward speed: {forward_speed}", flush=True)
            if should_log:
                logger.info(f"⬆️  Moving FORWARD {forward_speed:.0f} - Target too far")
        else:
            # Too close - move backward (or stop if very close)
            if result.distance < self.min_distance:
                forward_speed = 0
                if should_log:
                    logger.warning(f"⚠️  TOO CLOSE! Distance {result.distance:.2f}m < {self.min_distance:.2f}m - STOPPING")
            else:
                forward_speed = max(-self.base_speed * 0.5, -30)  # Limited reverse speed
                if should_log:
                    logger.info(f"⬇️  Moving BACKWARD {abs(forward_speed):.0f} - Target too close")

        # Steering based on tracking offset
        turn_amount = result.tracking_offset * self.turn_gain

        if should_log and abs(result.tracking_offset) > 0.05:
            direction = "LEFT ⬅️ " if result.tracking_offset < 0 else "RIGHT ➡️"
            logger.info(f"🔄 Turning {direction} (offset: {result.tracking_offset:+.3f}, turn: {turn_amount:+.0f})")

        # Differential drive
        left_speed = forward_speed - turn_amount
        right_speed = forward_speed + turn_amount

        # Clamp to motor limits
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        if should_log:
            logger.info(f"🎮 Motor Speeds: L={left_speed:+.0f}, R={right_speed:+.0f}")

        return (left_speed, right_speed)

    def process_vision_result(self, result: VisionResult):
        """
        Process vision result and update motor control

        Args:
            result: Current vision detection result
        """
        # Skip if motors not available
        if self.motors is None:
            return

        if result.found:
            self.last_detection_time = time.time()

            if self.tracking_enabled:
                # Calculate and apply motor speeds
                self.calculate_motor_speeds(result)
                print("turning the motors!!!", flush=True)
                #self.motors.set_motors(left_speed, right_speed)
                #time.sleep(0.5)  # Delay to allow motors to respond
        else:
            # No detection
            time_since_detection = time.time() - self.last_detection_time

            if time_since_detection > self.max_tracking_age:
                # Lost target - stop motors
                self.motors.stop()
                time.sleep(0.5)  # Delay to ensure motors stop

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
        if self.motors:
            self.motors.stop()
            self.motors.cleanup()
            print("✅ Motors stopped and cleaned up")
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
