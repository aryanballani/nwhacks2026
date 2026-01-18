"""
Grocery Buddy Vision System
Unified person tracking and object detection
"""

from .camera_controller import CameraController, CameraMode, VisionResult
from .aruco_tracker import ArucoTracker, ArucoDetection
from .object_detector import ObjectDetector, ObjectDetection
from . import config

__all__ = [
    "CameraController",
    "CameraMode",
    "VisionResult",
    "ArucoTracker",
    "ArucoDetection",
    "ObjectDetector",
    "ObjectDetection",
    "config"
]

__version__ = "1.0.0"
