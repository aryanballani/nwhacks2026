# Grocery Buddy - Raspberry Pi

Python-based robot control system with computer vision tracking.

## Features

- **Color-based tracking** using OpenCV
- **Motor control** via pigpio for smooth PWM
- **WebSocket server** for Android app communication
- **Ultrasonic sensor** for obstacle detection
- **Autonomous following** with distance maintenance

## Requirements

### Hardware
- Raspberry Pi 4/5 (4GB+ RAM recommended)
- Pi Camera v2 or USB Camera
- L298N Motor Driver
- 2x DC Motors (6-12V)
- HC-SR04 Ultrasonic Sensor
- Power supplies (for Pi and motors)

### Software
- Python 3.9+
- OpenCV
- pigpio
- websockets
- asyncio

## Installation

### For Laptop Testing (Linux/Mac)

```bash
# Install system dependencies
sudo apt install -y python3-opencv python3-numpy  # Ubuntu/Debian
# or
brew install opencv python-numpy  # Mac

# Optional: Install YOLO for advanced object detection
pip3 install ultralytics --user
# or use --break-system-packages if needed
```

### For Raspberry Pi Deployment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-opencv python3-numpy \
    python3-picamera2 libatlas-base-dev

# Install Python packages (use --break-system-packages if needed)
pip3 install flask flask-cors websockets asyncio RPi.GPIO ultralytics --break-system-packages

# Enable camera
sudo raspi-config
# Interface Options → Camera → Enable

# Enable hardware PWM (for GPIO control)
sudo pigpiod
```

## Project Structure

```
raspberry-pi/
├── main.py                 # Main robot controller entry point
├── test_vision.py          # Vision system testing script
├── requirements.txt        # Python dependencies
├── vision/                 # Unified vision system (NEW)
│   ├── __init__.py
│   ├── config.py           # Vision configuration & HSV ranges
│   ├── camera_controller.py # Main dual-mode coordinator
│   ├── person_tracker.py   # Color-based person following
│   └── object_detector.py  # YOLO + color grocery detection
├── motors/
│   ├── __init__.py
│   └── motor_controller.py # L298N motor control with GPIO mock
├── sensors/
│   ├── __init__.py
│   └── ultrasonic.py       # Distance sensor
└── server/
    ├── __init__.py
    └── websocket_server.py # WebSocket communication
```

## Configuration

Edit `config.py` to customize:
- GPIO pin assignments
- Target color HSV range
- Following distance
- Motor speeds
- Camera settings

## Running

### Testing Vision System Only (Laptop or Pi)

```bash
# Test the unified vision system with webcam
cd raspberry-pi
python3 test_vision.py

# Controls:
# M - Toggle mode (FOLLOW ↔ SCAN)
# C - Calibrate person marker (hold pink/magenta object in center)
# Q - Quit
```

### Running Full Robot Controller

```bash
# On Laptop (for testing, motors will be mocked)
cd raspberry-pi
python3 main.py

# On Raspberry Pi (with real motors)
sudo pigpiod  # Start GPIO daemon first
python3 main.py

# For headless operation (no display)
python3 main.py --headless

# Without YOLO (color detection only)
python3 main.py --no-yolo

# Controls (interactive mode):
# T - Toggle tracking ON/OFF
# M - Toggle mode (FOLLOW ↔ SCAN)
# C - Calibrate person marker
# E - Emergency stop
# Q - Quit
```

The WebSocket server will start on port 8765 (when implemented).

## Dual-Mode Vision System

The robot has two operating modes:

### FOLLOW Mode
- Tracks a person using a colored marker (pink/magenta/green)
- Maintains target distance of ~1 meter
- Requires calibration (press 'C' key)
- Uses HSV color detection for robust tracking

### SCAN Mode
- Detects grocery items using YOLO or color detection
- Supports: banana, apple, orange (expandable)
- Automatically falls back to color detection if YOLO unavailable
- Provides object labels and confidence scores

Switch modes anytime by pressing 'M' key.

## Calibration

### Person Tracking (FOLLOW Mode)
1. Start the robot in FOLLOW mode
2. Hold colored marker (pink/magenta) in center of camera view
3. Press 'C' to calibrate
4. The robot will lock onto your marker color
5. Press 'T' to enable tracking

### Object Detection (SCAN Mode)
- No calibration needed
- Show banana/apple/orange to camera
- System automatically detects with YOLO or color matching

## Color Marker Setup

Use a bright neon marker (15cm x 15cm minimum):
- **Pink/Magenta** (recommended) - HSV: (140-170, 100-255, 100-255)
- **Neon Green** - HSV: (35-85, 100-255, 100-255)
- **Neon Orange** - HSV: (5-25, 150-255, 150-255)

## Troubleshooting

### Motors not spinning
- Check L298N ENA/ENB jumpers
- Verify GPIO pin connections
- Check motor power supply voltage

### Camera not found
- Run `vcgencmd get_camera`
- Check ribbon cable connection
- Try `sudo raspi-config` to enable camera

### Poor tracking
- Adjust HSV color range in config.py
- Increase marker size
- Improve lighting conditions

### WebSocket disconnects
- Check WiFi signal strength
- Use static IP or `raspberrypi.local`
- Verify firewall settings

## API Commands

The WebSocket server accepts JSON commands:

```json
{"command": "calibrate"}
{"command": "start_tracking"}
{"command": "stop_tracking"}
{"command": "emergency_stop"}
{"command": "get_status"}
```

Status responses include:
- `tracking`: bool
- `target_locked`: bool
- `distance`: float (meters)
- `battery`: int (percentage)
- `obstacle_detected`: bool
