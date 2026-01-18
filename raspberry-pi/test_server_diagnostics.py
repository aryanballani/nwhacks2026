#!/usr/bin/env python3
"""
Server Diagnostics - Test motor and vision integration
Shows exactly what's happening when app sends commands
"""

import sys
import time
import logging
from main import RobotController
from vision import CameraMode

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_initialization():
    """Test robot initialization"""
    logger.info("=" * 70)
    logger.info("TEST 1: Robot Initialization")
    logger.info("=" * 70)

    try:
        robot = RobotController(camera_id=0, use_yolo=False)
        logger.info(f"✅ Robot initialized")
        logger.info(f"   Camera: {robot.camera is not None}")
        logger.info(f"   Motors: {robot.motors is not None}")
        logger.info(f"   Tracking enabled: {robot.tracking_enabled}")
        logger.info(f"   Emergency stop: {robot.emergency_stop}")
        return robot
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vision_follow_mode(robot):
    """Test vision in FOLLOW mode"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Vision - FOLLOW Mode (10 frames)")
    logger.info("=" * 70)

    robot.camera.set_mode(CameraMode.FOLLOW)
    time.sleep(0.5)  # Let mode switch stabilize

    detections = 0
    for i in range(10):
        frame, result = robot.camera.process_frame()
        if result.found:
            detections += 1
            logger.info(
                f"Frame {i+1}: ✓ ArUco detected - "
                f"Distance={result.distance:.2f}m, Offset={result.tracking_offset:+.3f}, "
                f"Calibrated={robot.camera.aruco_tracker.focal_length_px is not None}"
            )
        else:
            logger.info(f"Frame {i+1}: ⚪ No ArUco detected")
        time.sleep(0.1)

    logger.info(f"\nDetection rate: {detections}/10 frames")
    return detections > 0


def test_calibration(robot):
    """Test calibration"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Calibration")
    logger.info("=" * 70)
    logger.info("Hold ArUco marker at 30cm from camera (centered)...")

    # Wait for marker to be in position
    for i in range(5, 0, -1):
        logger.info(f"Starting in {i}...")
        time.sleep(1)

    frame, _ = robot.camera.process_frame()
    if frame is not None:
        success = robot.camera.calibrate_person_marker(frame)
        if success:
            logger.info("✅ Calibration successful!")
            logger.info(f"   Focal length: {robot.camera.aruco_tracker.focal_length_px:.1f}px")
            return True
        else:
            logger.error("❌ Calibration failed - is ArUco marker visible?")
            return False
    else:
        logger.error("❌ No frame available")
        return False


def test_motor_control(robot):
    """Test motor control"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Motor Control (Simulated)")
    logger.info("=" * 70)

    if robot.motors is None:
        logger.warning("⚠️  Motors not available - skipping motor test")
        return False

    # Simulate vision result
    from vision import VisionResult
    test_result = VisionResult(
        mode=CameraMode.FOLLOW,
        found=True,
        label="Test ArUco",
        confidence=1.0,
        center=(160, 120),  # Centered for 320x240
        bbox=(100, 80, 120, 80),
        distance=0.8,  # 0.8m - closer than target (1.0m)
        tracking_offset=0.2  # Slightly to the right
    )

    logger.info("Simulated detection:")
    logger.info(f"  Distance: {test_result.distance}m (target: 1.0m)")
    logger.info(f"  Offset: {test_result.tracking_offset:+.3f} (right)")

    # Enable tracking
    robot.tracking_enabled = True

    # Process result
    logger.info("\nCalculating motor speeds...")
    left, right = robot.calculate_motor_speeds(test_result)
    logger.info(f"  Left motor: {left:+.0f}")
    logger.info(f"  Right motor: {right:+.0f}")

    # Don't actually move motors in test
    logger.info("\n⚠️  Not actually moving motors in test mode")

    robot.tracking_enabled = False
    return True


def test_tracking_loop(robot):
    """Test tracking loop for 5 seconds"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Tracking Loop (5 seconds)")
    logger.info("=" * 70)

    if robot.motors is None:
        logger.warning("⚠️  Motors not available")
        logger.info("Will test vision loop only (no motor commands)")

    robot.camera.set_mode(CameraMode.FOLLOW)
    robot.tracking_enabled = True

    logger.info("✅ Tracking enabled - monitoring for 5 seconds...")
    start_time = time.time()
    frame_count = 0
    detections = 0

    while time.time() - start_time < 5.0:
        frame, result = robot.camera.process_frame()
        frame_count += 1

        if result.found:
            detections += 1

            # This is what happens in the broadcast loop
            robot.process_vision_result(result)

            if frame_count % 10 == 0:
                logger.info(
                    f"Frame {frame_count}: ArUco detected - "
                    f"Dist={result.distance:.2f}m, Offset={result.tracking_offset:+.3f}"
                )
        else:
            if frame_count % 10 == 0:
                logger.info(f"Frame {frame_count}: No detection")

        time.sleep(0.1)

    robot.tracking_enabled = False
    if robot.motors:
        robot.motors.stop()

    logger.info(f"\nResults:")
    logger.info(f"  Total frames: {frame_count}")
    logger.info(f"  Detections: {detections}")
    logger.info(f"  Detection rate: {detections/frame_count*100:.1f}%")

    return detections > 0


def main():
    """Run all diagnostic tests"""
    logger.info("╔═══════════════════════════════════════════════════════════════════╗")
    logger.info("║         GROCERY BUDDY - Server Diagnostics                       ║")
    logger.info("╚═══════════════════════════════════════════════════════════════════╝")

    try:
        # Test 1: Initialization
        robot = test_initialization()
        if robot is None:
            logger.error("\n❌ Cannot continue - initialization failed")
            return

        time.sleep(1)

        # Test 2: Vision
        vision_ok = test_vision_follow_mode(robot)
        time.sleep(1)

        # Test 3: Calibration (optional - user needs marker)
        input("\n📸 Press ENTER when ready to test calibration (or Ctrl+C to skip)...")
        calibration_ok = test_calibration(robot)
        time.sleep(1)

        # Test 4: Motor Control
        motor_ok = test_motor_control(robot)
        time.sleep(1)

        # Test 5: Tracking Loop
        input("\n🎯 Press ENTER to start 5-second tracking test (or Ctrl+C to skip)...")
        tracking_ok = test_tracking_loop(robot)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Initialization:  {'✅' if robot else '❌'}")
        logger.info(f"Vision Detection: {'✅' if vision_ok else '❌'}")
        logger.info(f"Calibration:     {'✅' if calibration_ok else '❌'}")
        logger.info(f"Motor Control:   {'✅' if motor_ok else '❌'}")
        logger.info(f"Tracking Loop:   {'✅' if tracking_ok else '❌'}")

        if robot.motors is None:
            logger.warning("\n⚠️  WARNING: Motors not initialized!")
            logger.warning("This could be why motors don't move in the app.")
            logger.warning("Check GPIO permissions and motor controller setup.")

        if not vision_ok:
            logger.warning("\n⚠️  WARNING: ArUco detection not working!")
            logger.warning("Make sure ArUco marker (5x5_50, 5cm x 5cm) is visible.")

        robot.shutdown()

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("\n✅ Diagnostics complete")


if __name__ == "__main__":
    main()
