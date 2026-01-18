"""
Motor control using RPi.GPIO for PWM control
Controls differential drive robot base via L298N motor driver
"""

import sys

# Try to import GPIO, fallback to mock for testing on laptop
try:
    import RPi.GPIO as GPIO
    import time
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    print("⚠ RPi.GPIO not available - using mock mode for testing")
    GPIO_AVAILABLE = False

    # Mock GPIO for laptop testing
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0

        @staticmethod
        def setmode(mode): pass

        @staticmethod
        def setup(pin, mode): pass

        @staticmethod
        def output(pin, value): pass

        @staticmethod
        def PWM(pin, freq):
            class MockPWM:
                def start(self, duty): pass
                def ChangeDutyCycle(self, duty): pass
                def stop(self): pass
            return MockPWM()

        @staticmethod
        def cleanup(): pass

    GPIO = MockGPIO()


# L298N Motor Driver Pin Configuration
# Motor A (Left)
ENA = 12  # PWM pin for left motor speed
IN1 = 17  # Direction control 1 for left motor
IN2 = 27  # Direction control 2 for left motor

# Motor B (Right)
ENB = 13  # PWM pin for right motor speed
IN3 = 22  # Direction control 1 for right motor
IN4 = 23  # Direction control 2 for right motor

PWM_FREQUENCY = 1000  # 1kHz PWM frequency

class MotorController:
    """Controls robot motors via L298N motor driver"""

    def __init__(self):
        """Initialize GPIO pins and PWM"""
        self.enabled = GPIO_AVAILABLE

        if not self.enabled:
            print("✓ MotorController initialized (MOCK MODE - no GPIO)")
            return

        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(ENA, GPIO.OUT)
            GPIO.setup(IN1, GPIO.OUT)
            GPIO.setup(IN2, GPIO.OUT)
            GPIO.setup(ENB, GPIO.OUT)
            GPIO.setup(IN3, GPIO.OUT)
            GPIO.setup(IN4, GPIO.OUT)

            # Initialize PWM
            self.pwm_left = GPIO.PWM(ENA, PWM_FREQUENCY)
            self.pwm_right = GPIO.PWM(ENB, PWM_FREQUENCY)

            # Start PWM with 0% duty cycle (motors off)
            self.pwm_left.start(0)
            self.pwm_right.start(0)

            print("✓ MotorController initialized (GPIO mode)")
        except Exception as e:
            print(f"⚠ GPIO initialization failed: {e}")
            print("✓ Motor Controller running in MOCK MODE")
            self.enabled = False

    def set_motors(self, left_speed: float, right_speed: float):
        """
        Set motor speeds with direction

        Args:
            left_speed: Speed from -100 (full reverse) to 100 (full forward)
            right_speed: Speed from -100 (full reverse) to 100 (full forward)
        """
        # Clamp speeds
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        if not self.enabled:
            # Mock mode - just print
            if abs(left_speed) > 5 or abs(right_speed) > 5:
                print(f"🤖 Motors: L={left_speed:+.0f}% R={right_speed:+.0f}%")
            return

        # Left motor
        if left_speed >= 0:
            self.pwm_left.ChangeDutyCycle(abs(left_speed))
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
        else:
            self.pwm_left.ChangeDutyCycle(abs(left_speed))
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)

        # Right motor
        if right_speed >= 0:
            self.pwm_right.ChangeDutyCycle(abs(right_speed))
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
        else:
            self.pwm_right.ChangeDutyCycle(abs(right_speed))
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)

    def forward(self, speed: float = 100):
        """Move forward at specified speed"""
        self.set_motors(speed, speed)

    def backward(self, speed: float = 100):
        """Move backward at specified speed"""
        self.set_motors(-speed, -speed)

    def turn_right(self, speed_fast: float = 70, speed_slow: float = 30):
        """Turn right in place"""
        self.set_motors(speed_fast, speed_slow)

    def turn_left(self, speed_fast: float = 70, speed_slow: float = 30):
        """Turn left in place"""
        self.set_motors(speed_slow, speed_fast)

    def stop(self):
        """Stop all motors"""
        self.set_motors(0, 0)

    def cleanup(self):
        """Cleanup GPIO resources"""
        self.stop()
        if self.enabled:
            self.pwm_left.stop()
            self.pwm_right.stop()
            GPIO.cleanup()
        print("✓ MotorController cleanup complete")
