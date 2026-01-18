#!/usr/bin/env python3
"""
Grocery Buddy - Main Server with WebSocket Integration
Run this on Raspberry Pi for Android app control
"""

import sys
import asyncio
from main import RobotController
from server.websocket_server import RobotWebSocketServer


async def main_async():
    """Main async entry point"""
    print("=" * 70)
    print("GROCERY BUDDY - WebSocket Server Mode")
    print("=" * 70)
    print()

    robot = None
    try:
        # Initialize robot controller with camera and object detection
        robot = RobotController(camera_id=0, use_yolo=True)

        # Initialize WebSocket server
        server = RobotWebSocketServer(robot)

        print("\n🌐 Starting WebSocket server on 0.0.0.0:8765")
        print("📱 Connect your Android app to this Pi's IP address")
        print("⚠️  Press Ctrl+C to stop\n")

        # Start server (this will block and run the server)
        await server.start(host="0.0.0.0", port=8765)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    asyncio.run(main_async())
