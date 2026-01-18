#!/usr/bin/env python3
"""
Motor receiver - Receives vision data and controls motors
Maintains target distance from ArUco marker
"""

import socket
import RPi.GPIO as GPIO
import time
import logging
from motors.motor_controller import MotorController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Network settings
HOST = ""  # Listen on all interfaces
PORT = 5000

# Control parameters
TARGET_DISTANCE = 1.0  # Target distance in meters
DISTANCE_TOLERANCE = 0.2  # Acceptable distance range (±0.2m)
MIN_DISTANCE = 0.3  # Safety minimum distance
MAX_DISTANCE = 3.0  # Maximum following distance

# Speed settings
BASE_SPEED = 60  # Base forward/backward speed (0-100)
MAX_SPEED = 90   # Maximum speed
MIN_SPEED = 30   # Minimum speed for movement

# Turning settings
TURN_GAIN = 80  # Steering sensitivity (higher = sharper turns)
DEADZONE = 0.05  # Offset deadzone to ignore small deviations

# Motor controller instance
motor = None


def calculate_motor_speeds(offset: float, distance: float) -> tuple:
    """
    Calculate left and right motor speeds based on vision data

    Args:
        offset: Horizontal offset from center (-1.0 to +1.0)
                Negative = target is on LEFT
                Positive = target is on RIGHT
        distance: Distance to target in meters

    Returns:
        (left_speed, right_speed) tuple
    """

    # Distance-based forward/backward speed
    distance_error = distance - TARGET_DISTANCE

    logger.info(f"📏 Distance: {distance:.2f}m (target: {TARGET_DISTANCE:.2f}m, error: {distance_error:+.2f}m)")

    # Determine forward/backward speed
    if distance < MIN_DISTANCE:
        # Too close - STOP for safety
        forward_speed = 0
        logger.warning(f"⚠️  TOO CLOSE! Distance: {distance:.2f}m < {MIN_DISTANCE:.2f}m - STOPPING")
    elif distance > MAX_DISTANCE:
        # Too far - stop following
        forward_speed = 0
        logger.warning(f"⚠️  TOO FAR! Distance: {distance:.2f}m > {MAX_DISTANCE:.2f}m - STOPPING")
    elif abs(distance_error) < DISTANCE_TOLERANCE:
        # Within acceptable range - maintain position (slow adjustment)
        forward_speed = distance_error * 20  # Gentle adjustment
        logger.info(f"✓ Distance OK - Minor adjustment: {forward_speed:.0f}")
    elif distance_error > 0:
        # Too far - move forward
        # Scale speed based on how far we need to go
        speed_factor = min(distance_error / TARGET_DISTANCE, 1.0)
        forward_speed = BASE_SPEED + (MAX_SPEED - BASE_SPEED) * speed_factor
        logger.info(f"⬆️  FORWARD {forward_speed:.0f} - Target too far")
    else:
        # Too close - move backward
        speed_factor = min(abs(distance_error) / TARGET_DISTANCE, 1.0)
        forward_speed = -(MIN_SPEED + (BASE_SPEED - MIN_SPEED) * speed_factor)
        logger.info(f"⬇️  BACKWARD {abs(forward_speed):.0f} - Target too close")

    # Apply deadzone to offset (ignore tiny deviations)
    if abs(offset) < DEADZONE:
        effective_offset = 0.0
        logger.debug(f"Offset {offset:+.3f} within deadzone, treating as centered")
    else:
        effective_offset = offset

    # Calculate turn amount
    # Negative offset = target on LEFT, need to turn LEFT (reduce left motor, increase right)
    # Positive offset = target on RIGHT, need to turn RIGHT (increase left motor, reduce right)
    turn_amount = effective_offset * TURN_GAIN

    if abs(effective_offset) > DEADZONE:
        direction = "LEFT ⬅️ " if offset < 0 else "RIGHT ➡️"
        logger.info(f"🔄 Turn {direction} (offset: {offset:+.3f}, turn: {turn_amount:+.0f})")

    # Apply differential steering
    # Turn LEFT: reduce left motor, keep/increase right motor
    # Turn RIGHT: reduce right motor, keep/increase left motor
    left_speed = forward_speed - turn_amount
    right_speed = forward_speed + turn_amount

    # Clamp to motor limits
    left_speed = max(-100, min(100, left_speed))
    right_speed = max(-100, min(100, right_speed))

    logger.info(f"🎮 Motor speeds: L={left_speed:+.0f}, R={right_speed:+.0f}")

    return (left_speed, right_speed)


def control_motors(offset: float, distance: float):
    """
    Control motors based on vision data
    """
    if motor is None:
        logger.error("Motor controller not initialized!")
        return

    # Calculate motor speeds
    left_speed, right_speed = calculate_motor_speeds(offset, distance)

    # Apply to motors
    if abs(left_speed) < 5 and abs(right_speed) < 5:
        # Speeds too low, just stop
        motor.stop()
        logger.info("⏸️  Motors STOPPED (speeds too low)")
    else:
        motor.set_motors(left_speed, right_speed)
        logger.info(f"✓ Motors set: L={left_speed:+.0f}, R={right_speed:+.0f}")


def main():
    """Main receiver loop"""
    global motor

    logger.info("=" * 70)
    logger.info("MOTOR RECEIVER - Vision-based motor control")
    logger.info("=" * 70)
    logger.info(f"Target distance: {TARGET_DISTANCE}m (±{DISTANCE_TOLERANCE}m)")
    logger.info(f"Safety limits: {MIN_DISTANCE}m - {MAX_DISTANCE}m")
    logger.info(f"Base speed: {BASE_SPEED}, Max speed: {MAX_SPEED}")
    logger.info(f"Turn gain: {TURN_GAIN}, Deadzone: {DEADZONE}")
    logger.info("=" * 70)

    # Initialize motor controller
    try:
        motor = MotorController()
        logger.info("✓ Motor controller initialized")
    except Exception as e:
        logger.error(f"Failed to initialize motors: {e}")
        return

    # Setup socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)

    logger.info(f"🔌 Listening on port {PORT}...")
    logger.info("Waiting for connection from vision system...")

    try:
        while True:
            conn, addr = sock.accept()
            logger.info(f"✓ Connected: {addr}")

            try:
                while True:
                    data = conn.recv(1024).decode().strip()
                    if not data:
                        logger.warning("Connection closed by client")
                        break

                    logger.debug(f"📥 Received: {data}")

                    try:
                        # Parse "offset distance" format
                        parts = data.split()
                        if len(parts) != 2:
                            logger.error(f"Invalid data format: {data}")
                            continue

                        offset = float(parts[0])
                        distance = float(parts[1])

                        logger.info(f"\n{'='*60}")
                        logger.info(f"📊 Vision Data: Offset={offset:+.3f}, Distance={distance:.2f}m")
                        logger.info(f"{'='*60}")

                        # Control motors
                        control_motors(offset, distance)

                    except ValueError as e:
                        logger.error(f"Failed to parse data: {data} - {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing data: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            except socket.error as e:
                logger.error(f"Socket error: {e}")
            finally:
                conn.close()
                motor.stop()
                logger.info("⏹️  Connection closed, motors stopped")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    finally:
        motor.stop()
        motor.cleanup()
        sock.close()
        GPIO.cleanup()
        logger.info("✓ Cleanup complete")


if __name__ == "__main__":
    main()
