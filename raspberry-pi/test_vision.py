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
    print("  C     - Calibrate person marker (FOLLOW mode only)")
    print("  Q/ESC - Quit")
    print("  SPACE - Take screenshot")
    print("\n📹 MODES:")
    print("  FOLLOW - Track person with colored marker (pink/magenta/green)")
    print("  SCAN   - Detect grocery items (banana, apple, orange, etc.)")
    print("\n💡 TIPS:")
    print("  - For FOLLOW mode: Hold pink/magenta object in center, press 'C'")
    print("  - For SCAN mode: Show banana/apple/orange to camera")
    print("  - YOLO will be used if installed (pip install ultralytics)")
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
                    print(
                        f"\r✓ {result.label} | "
                        f"Conf: {result.confidence*100:.0f}% | "
                        f"Dist: {result.distance:.1f}m | "
                        f"Offset: {result.tracking_offset:+.2f} | "
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

            if key == ord('q') or key == 27:  # Q or ESC
                break

            elif key == ord('m'):
                # Toggle mode
                new_mode = CameraMode.SCAN if camera.mode == CameraMode.FOLLOW else CameraMode.FOLLOW
                camera.set_mode(new_mode)
                print(f"\n\n🔄 Mode switched to: {new_mode.value.upper()}\n")

            elif key == ord('c'):
                # Calibrate (FOLLOW mode only)
                if camera.mode == CameraMode.FOLLOW:
                    print("\n\n📸 Calibrating... Hold marker in center of frame...")
                    success = camera.calibrate_person_marker(frame)
                    if success:
                        print("✅ Calibration successful!\n")
                    else:
                        print("❌ Calibration failed\n")
                else:
                    print("\n\n⚠️  Calibration only works in FOLLOW mode (press 'M' to switch)\n")

            elif key == ord(' '):
                # Screenshot
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
