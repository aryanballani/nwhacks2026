#!/usr/bin/env python3
"""
Test script for threaded camera controller
"""
import traceback
import time
from vision import CameraController

try:
    print("Initializing camera controller...")
    cam = CameraController(camera_id=0, use_yolo=True, threaded=True)

    print("Testing frame processing for 5 seconds...")
    for i in range(50):  # 50 frames, ~5 seconds
        frame, result = cam.process_frame()
        if frame is not None:
            print(f"Frame {i}: {result.mode.value} - Found: {result.found}, Label: {result.label}")
        else:
            print(f"Frame {i}: No frame")
        time.sleep(0.1)

    print("\nTest completed successfully!")
    cam.release()

except Exception as e:
    print(f"\nError: {e}")
    traceback.print_exc()
