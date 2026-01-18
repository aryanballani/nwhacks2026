#!/usr/bin/env python3
"""
Test script for motor control with distance tracking
Shows detailed logs of ArUco detection, distance, and motor commands
"""

import time
import logging
from vision import CameraController, CameraMode

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Control parameters
TARGET_DISTANCE = 1.0  # meters
DISTANCE_TOLERANCE = 0.2  # meters

def calculate_motor_action(offset: float, distance: float, calibrated: bool):
    """
    Calculate what motor action should be taken
    Returns human-readable description
    """
    if not calibrated:
        return "⚠️  NOT CALIBRATED - No distance available"

    distance_error = distance - TARGET_DISTANCE

    # Determine forward/backward action
    if abs(distance_error) < DISTANCE_TOLERANCE:
        distance_action = f"✓ MAINTAIN (distance OK: {distance:.2f}m)"
    elif distance_error > 0:
        distance_action = f"⬆️  FORWARD (too far: {distance:.2f}m > {TARGET_DISTANCE}m)"
    else:
        distance_action = f"⬇️  BACKWARD (too close: {distance:.2f}m < {TARGET_DISTANCE}m)"

    # Determine turning action
    if abs(offset) < 0.05:
        turn_action = "↕️  STRAIGHT (centered)"
    elif offset < 0:
        turn_action = f"⬅️  TURN LEFT (offset: {offset:+.3f})"
    else:
        turn_action = f"➡️  TURN RIGHT (offset: {offset:+.3f})"

    return f"{distance_action} + {turn_action}"


def main():
    logger.info("=" * 70)
    logger.info("MOTOR DISTANCE TEST - ArUco tracking with motor control visualization")
    logger.info("=" * 70)
    logger.info(f"Target distance: {TARGET_DISTANCE}m (±{DISTANCE_TOLERANCE}m)")
    logger.info("")
    logger.info("Instructions:")
    logger.info("1. Switch to FOLLOW mode (should auto-switch)")
    logger.info("2. Hold ArUco marker in center of camera")
    logger.info("3. The system will tell you to calibrate if needed")
    logger.info("4. Move marker closer/farther to see distance changes")
    logger.info("5. Move marker left/right to see offset changes")
    logger.info("=" * 70)

    try:
        # Initialize camera in FOLLOW mode
        logger.info("Initializing camera controller...")
        cam = CameraController(camera_id=0, use_yolo=False, threaded=True)

        logger.info("Switching to FOLLOW mode...")
        cam.set_mode(CameraMode.FOLLOW)

        time.sleep(1)  # Let threads stabilize

        logger.info("\n✓ Camera ready! Starting detection loop...\n")

        frame_count = 0
        last_calibration_reminder = 0

        while True:
            frame, result = cam.process_frame()

            if frame is None:
                continue

            frame_count += 1

            # Check if we found ArUco marker
            if result.found:
                calibrated = cam.aruco_tracker.focal_length_px is not None

                # Get distance (will be 0.0 if not calibrated)
                distance = result.distance
                offset = result.tracking_offset

                # Calculate motor action
                action = calculate_motor_action(offset, distance, calibrated)

                # Print status every 10 frames (about once per second at 10 fps)
                if frame_count % 10 == 0:
                    logger.info("")
                    logger.info("─" * 70)
                    logger.info(f"📊 ArUco Detection #{frame_count}")
                    logger.info(f"   Calibrated: {calibrated}")
                    logger.info(f"   Distance: {distance:.2f}m")
                    logger.info(f"   Offset: {offset:+.3f} ({'left' if offset < 0 else 'right' if offset > 0 else 'center'})")
                    logger.info(f"   Motor Action: {action}")
                    logger.info("─" * 70)

            else:
                # No ArUco detected
                if frame_count % 30 == 0:
                    logger.warning("⚠️  No ArUco marker detected - hold marker in view")

            # Remind about calibration every 5 seconds if not calibrated
            if not cam.aruco_tracker.focal_length_px:
                if frame_count - last_calibration_reminder > 50:
                    logger.warning("")
                    logger.warning("╔═════════════════════════════════════════════════════════╗")
                    logger.warning("║  ⚠️  CALIBRATION REQUIRED!                              ║")
                    logger.warning("║                                                         ║")
                    logger.warning("║  1. Hold ArUco marker at 30cm from camera              ║")
                    logger.warning("║  2. Press 'C' in the app to calibrate                  ║")
                    logger.warning("║  3. Distance measurement will then be available        ║")
                    logger.warning("╚═════════════════════════════════════════════════════════╝")
                    logger.warning("")
                    last_calibration_reminder = frame_count

            time.sleep(0.1)  # ~10 fps

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cam.release()
        logger.info("\n✓ Test complete")


if __name__ == "__main__":
    main()
