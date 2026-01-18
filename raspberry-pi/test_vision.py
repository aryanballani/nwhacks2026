#!/usr/bin/env python3
"""
Test script for unified vision system
Works on laptop and Raspberry Pi
"""

import cv2
import time
import sys
from vision import CameraController, CameraMode


def main():
    print("=" * 70)
    print("GROCERY BUDDY - Unified Vision System Test")
    print("=" * 70)
    print("\n🎮 CONTROLS:")
    print("  M     - Toggle mode (FOLLOW ↔ SCAN)")
    print("  C     - Calibrate ArUco marker distance (FOLLOW mode only)")
    print("  Q/ESC - Quit")
    print("  SPACE - Take screenshot")
    print("\n📹 MODES:")
    print("  FOLLOW - Track ArUco marker (printed QR-style fiducial)")
    print("  SCAN   - Detect grocery items (banana, apple, orange, etc.)")
    print("\n💡 TIPS:")
    print("  - For FOLLOW mode: Hold ArUco marker in center, press 'C'")
    print("  - For SCAN mode: Show banana/apple/orange to camera")
    print("  - ONNX will be used if models/best.onnx is present (pip install onnxruntime)")
    print("=" * 70)
    print()

    try:
        # Initialize camera controller
        # Will try to use YOLO, fallback to color detection if not available
        camera = CameraController(camera_id=0, use_yolo=True)

        print("\n✅ Camera initialized successfully!")
        print(f"📷 Resolution: {camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"🎯 Current mode: {camera.mode.value.upper()}")
        print("\nPress any key in the video window to start...\n")

        frame_count = 0
        fps_time = time.time()

        while True:
            # Process frame
            frame, result = camera.process_frame()

            if frame is None:
                print("❌ Failed to read frame")
                break

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - fps_time) if fps_time else 0
                fps_time = now

                # Print detection info
                if result.found:
                    print(f"\r{'  ' * 10}", end='')  # Clear line
                    dist_text = f"{result.distance:.1f}m"
                    if result.mode == CameraMode.FOLLOW and result.raw_detection is not None:
                        distance = getattr(result.raw_detection, "distance", None)
                        if distance is None:
                            dist_text = "--"
                    coord_text = ""
                    if result.mode == CameraMode.FOLLOW:
                        locked_center = camera.aruco_tracker.get_locked_center(frame, normalized=True)
                        if locked_center is not None:
                            coord_text = f" | X: {locked_center[0]:+.2f} Y: {locked_center[1]:+.2f}"
                    print(
                        f"\r✓ {result.label} | "
                        f"Conf: {result.confidence*100:.0f}% | "
                        f"Dist: {dist_text} | "
                        f"Offset: {result.tracking_offset:+.2f}{coord_text} | "
                        f"FPS: {fps:.1f}",
                        end=''
                    )
                else:
                    print(f"\r{'  ' * 10}", end='')  # Clear line
                    print(f"\r⚪ {result.label} | FPS: {fps:.1f}", end='')

            # Display frame
            cv2.imshow("Grocery Buddy Vision System", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Normalize key to a lowercase character when possible so both
            # uppercase and lowercase inputs work (e.g. 'M' and 'm').
            k = None
            if key != 255:
                try:
                    k = chr(key).lower()
                except Exception:
                    k = None

            # Quit: 'q' or ESC
            if key == 27 or k == 'q':
                break

            # Toggle mode: 'm' (case-insensitive)
            elif k == 'm':
                new_mode = CameraMode.SCAN if camera.mode == CameraMode.FOLLOW else CameraMode.FOLLOW
                camera.set_mode(new_mode)
                print(f"\n\n🔄 Mode switched to: {new_mode.value.upper()}\n")

            # Calibrate: 'c' (case-insensitive)
            elif k == 'c':
                if camera.mode == CameraMode.FOLLOW:
                    print("\n\n📸 Calibrating... Hold marker in center of frame...")
                    success = camera.calibrate_person_marker(frame)
                    if success:
                        print("✅ Calibration successful!\n")
                    else:
                        print("❌ Calibration failed\n")
                else:
                    print("\n\n⚠️  Calibration only works in FOLLOW mode (press 'M' to switch)\n")

            # Screenshot: spacebar
            elif key == ord(' '):
                timestamp = int(time.time())
                filename = f"vision_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n\n📸 Screenshot saved: {filename}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n\n🛑 Shutting down...")
        camera.release()
        print("✅ Camera released")
        print("\nTest complete! 👋\n")


if __name__ == "__main__":
    main()
