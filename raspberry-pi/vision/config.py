"""
Vision System Configuration
HSV color ranges, YOLO settings, camera parameters
"""

import numpy as np

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ArUco Tracking
ARUCO_MARKER_LENGTH_CM = 5.0
ARUCO_CALIBRATION_DISTANCE_CM = 30.0

# Person Tracking - Color-based detection
PERSON_MARKER_COLORS = {
    "pink_magenta": {
        "lower": np.array([140, 100, 100]),
        "upper": np.array([170, 255, 255])
    },
    "neon_green": {
        "lower": np.array([35, 100, 100]),
        "upper": np.array([85, 255, 255])
    },
    "neon_orange": {
        "lower": np.array([5, 150, 150]),
        "upper": np.array([25, 255, 255])
    }
}

# Default person marker color
DEFAULT_MARKER = "pink_magenta"

# Object Detection - Grocery Items (Color-based fallback)
GROCERY_ITEM_COLORS = {
    "banana": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255])
    },
    "orange": {
        "lower": np.array([10, 100, 100]),
        "upper": np.array([20, 255, 255])
    },
    "apple_red": {
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 255])
    },
    "apple_green": {
        "lower": np.array([35, 50, 50]),
        "upper": np.array([85, 255, 255])
    },
    "tomato": {
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 255])
    },
    "lemon": {
        "lower": np.array([22, 100, 100]),
        "upper": np.array([32, 255, 255])
    },
    "broccoli": {
        "lower": np.array([40, 40, 40]),
        "upper": np.array([80, 255, 255])
    }
}

# YOLO Configuration (Ultralytics .pt)
YOLO_MODEL = "models/yolov8n.pt"  # Nano model for speed
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

# ONNX Configuration (Ultralytics export)
ONNX_MODEL = "models/best.onnx"
ONNX_INPUT_SIZE = 416
ONNX_CONFIDENCE_THRESHOLD = 0.4
ONNX_IOU_THRESHOLD = 0.45

# Class names for ONNX custom model (index-aligned)
# Example: ["banana", "apple", "orange"]
ONNX_CLASS_NAMES = []

# Grocery categories from YOLO COCO dataset
GROCERY_CLASSES = {
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    # Add more as needed
}

# Detection Parameters
MIN_CONTOUR_AREA = 500  # Minimum pixels for valid detection
MIN_OBJECT_AREA = 1000  # Minimum pixels for object detection

# Distance Estimation (based on area)
DISTANCE_CALIBRATION = {
    "very_close": (15000, float('inf')),  # > 15000 px² = 0.5m
    "close": (5000, 15000),                # 5000-15000 px² = 1.0m
    "medium": (1000, 5000),                # 1000-5000 px² = 2.0m
    "far": (0, 1000)                       # < 1000 px² = 3.0m
}

def estimate_distance(area: float) -> float:
    """Estimate distance in meters based on detected area"""
    if area >= DISTANCE_CALIBRATION["very_close"][0]:
        return 0.5
    elif area >= DISTANCE_CALIBRATION["close"][0]:
        return 1.0
    elif area >= DISTANCE_CALIBRATION["medium"][0]:
        return 2.0
    else:
        return 3.0
