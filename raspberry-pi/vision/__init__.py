"""
Grocery Buddy Vision System
Unified person tracking and object detection
"""

from .camera_controller import CameraController, CameraMode, VisionResult
from .person_tracker import PersonTracker, PersonDetection
from .object_detector import ObjectDetector, ObjectDetection
from . import config

__all__ = [
    "CameraController",
    "CameraMode",
    "VisionResult",
    "PersonTracker",
    "PersonDetection",
    "ObjectDetector",
    "ObjectDetection",
    "config"
]

__version__ = "1.0.0"
