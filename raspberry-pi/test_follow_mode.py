#!/usr/bin/env python3
"""
Test FOLLOW mode with ArUco markers
"""
import time
from vision import CameraController, CameraMode

try:
    print("Initializing camera controller...")
    cam = CameraController(camera_id=0, use_yolo=False, threaded=True)

    print("Switching to FOLLOW mode...")
    cam.set_mode(CameraMode.FOLLOW)

    print("\nTesting ArUco detection for 5 seconds...")
    print("Hold an ArUco marker (5x5_50 dictionary) in front of camera\n")

    for i in range(50):
        frame, result = cam.process_frame()
        if frame is not None and result.found:
            print(f"✓ ArUco detected! Offset: {result.tracking_offset:+.3f}, Distance: {result.distance:.2f}m")
        elif i % 10 == 0:
            print(f"⚪ No ArUco marker detected (frame {i})")
        time.sleep(0.1)

    print("\nTest completed!")
    cam.release()

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
