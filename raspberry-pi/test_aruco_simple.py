#!/usr/bin/env python3
"""
Simple ArUco detection test
Shows if camera can detect ANY ArUco markers
"""

import cv2
import numpy as np
import time

# Try different ArUco dictionaries
DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "7x7_50": cv2.aruco.DICT_7X7_50,
}

def test_aruco_detection():
    """Test ArUco detection with all common dictionaries"""
    print("=" * 70)
    print("ARUCO MARKER DETECTION TEST")
    print("=" * 70)
    print("\nTesting camera and ArUco detection...")
    print("Hold an ArUco marker in front of the camera\n")

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    print("✅ Camera opened")
    print("Testing detection for 10 seconds...\n")

    start_time = time.time()
    frame_count = 0
    detections = {}

    for dict_name in DICTIONARIES:
        detections[dict_name] = 0

    while time.time() - start_time < 10.0:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try each dictionary
        for dict_name, dict_type in DICTIONARIES.items():
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

            try:
                if hasattr(cv2.aruco, 'ArucoDetector'):
                    detector = cv2.aruco.ArucoDetector(aruco_dict)
                    corners, ids, _ = detector.detectMarkers(gray)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

                if ids is not None and len(ids) > 0:
                    detections[dict_name] += 1

                    if frame_count % 10 == 0:  # Print every 10 frames
                        print(f"✓ {dict_name}: Detected marker ID {ids[0][0]}")

            except Exception as e:
                pass

        time.sleep(0.1)

    cap.release()

    # Results
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print(f"Total frames tested: {frame_count}")
    print("\nDetections by dictionary:")

    found_any = False
    for dict_name, count in detections.items():
        percentage = (count / frame_count * 100) if frame_count > 0 else 0
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {dict_name:12s}: {count:3d} detections ({percentage:5.1f}%)")
        if count > 0:
            found_any = True

    if not found_any:
        print("\n❌ NO ARUCO MARKERS DETECTED!")
        print("\nPossible issues:")
        print("  1. No ArUco marker in camera view")
        print("  2. Marker too far from camera")
        print("  3. Poor lighting")
        print("  4. Marker is blurry or damaged")
        print("  5. Wrong marker type (try different dictionaries)")
        print("\nTry:")
        print("  - Move marker closer to camera")
        print("  - Ensure good lighting")
        print("  - Use a freshly printed marker")
        print("  - Generate marker from: https://chev.me/arucogen/")
    else:
        print("\n✅ ArUco detection is WORKING!")
        print("\nRecommended dictionary for this project: 5x5_50")
        if detections["5x5_50"] > 0:
            print("✅ 5x5_50 is being detected - you're all set!")
        else:
            print("⚠️  5x5_50 not detected - you may have wrong marker type")
            print("   Generate a 5x5_50 marker from: https://chev.me/arucogen/")


if __name__ == "__main__":
    test_aruco_detection()
