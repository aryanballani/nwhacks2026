"""
WebSocket server for Android app communication
Handles commands and broadcasts robot status
"""

import asyncio
import json
import websockets
import logging
import numpy as np
from typing import Set, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


class RobotWebSocketServer:
    """WebSocket server for robot control"""

    def __init__(self, robot_controller):
        """Initialize server with robot controller reference"""
        self.robot = robot_controller
        self.clients: Set = set()
        self.running = False

        # Camera feed settings
        self.stream_video = False
        self.video_quality = 30  # JPEG quality 0-100 (reduced from 50 for Pi performance)

    async def handler(self, websocket):
        """Handle client connections and messages"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")

        # Register client
        self.clients.add(websocket)

        try:
            # Send initial status
            await self.send_status(websocket)

            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_command(data, websocket)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {client_id}: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        finally:
            self.clients.discard(websocket)

    async def handle_command(self, data: dict, websocket):
        """Process commands from Android app"""
        command = data.get("command")
        logger.info(f"Received command: {command}")

        if command == "calibrate":
            # Calibrate person marker - try multiple times
            logger.info("📸 Starting calibration (trying up to 10 frames)...")
            success = False

            for attempt in range(10):
                frame, result = self.robot.camera.process_frame()

                if frame is not None and result.found:
                    logger.info(f"Attempt {attempt+1}: ArUco detected, calibrating...")
                    success = self.robot.camera.calibrate_person_marker(frame)

                    if success:
                        logger.info(f"✅ Calibration successful on attempt {attempt+1}!")
                        break
                    else:
                        logger.warning(f"Attempt {attempt+1}: Calibration failed, retrying...")
                else:
                    logger.warning(f"Attempt {attempt+1}: No ArUco in frame")

                await asyncio.sleep(0.1)  # Wait 100ms between attempts

            if not success:
                logger.error("❌ Calibration failed after 10 attempts")

            response = {
                "type": "calibration_result",
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))

        elif command == "start_tracking":
            # Enable tracking
            self.robot.tracking_enabled = True
            if self.robot.emergency_stop:
                self.robot.emergency_stop = False
            logger.info("✅ TRACKING ENABLED - Motors will now respond to vision")

        elif command == "stop_tracking":
            # Disable tracking
            self.robot.tracking_enabled = False
            if self.robot.motors:
                self.robot.motors.stop()
            logger.info("⏹️  Tracking disabled - Motors stopped")

        elif command == "emergency_stop":
            # Emergency stop - stop motors and disable tracking
            self.robot.emergency_stop = True
            self.robot.tracking_enabled = False
            if self.robot.motors:
                self.robot.motors.stop()
            logger.warning("🚨 EMERGENCY STOP activated")

        elif command == "set_mode":
            # Switch between FOLLOW and SCAN modes
            mode_str = data.get("mode", "scan").lower()
            from vision import CameraMode
            new_mode = CameraMode.FOLLOW if mode_str == "follow" else CameraMode.SCAN
            self.robot.camera.set_mode(new_mode)
            logger.info(f"Mode changed to: {new_mode.value}")

        elif command == "get_status":
            # Send current status
            await self.send_status(websocket)

        elif command == "start_video_stream":
            # Enable video streaming
            self.stream_video = True
            logger.info("Video streaming enabled")

        elif command == "stop_video_stream":
            # Disable video streaming
            self.stream_video = False
            logger.info("Video streaming disabled")

        else:
            logger.warning(f"Unknown command: {command}")

    async def send_status(self, websocket):
        """Send current robot status to a client"""
        try:
            # Get latest vision result
            frame, result = self.robot.camera.process_frame()

            # Get normalized x, y coordinates from the result
            x_offset, y_offset = 0.0, 0.0
            if result.mode.value == "follow" and result.found and result.center:
                # Calculate normalized offset from center (already computed in vision result)
                h, w = frame.shape[:2] if frame is not None else (480, 640)
                x_offset = float((int(result.center[0]) - w / 2) / (w / 2))
                y_offset = float((int(result.center[1]) - h / 2) / (h / 2))

            status = {
                "type": "status",
                "tracking": bool(self.robot.tracking_enabled),
                "emergency_stop": bool(self.robot.emergency_stop),
                "target_locked": bool(result.found),
                "distance": float(result.distance) if result.found else 0.0,
                "mode": str(self.robot.camera.mode.value),
                "calibrated": self.robot.camera.aruco_tracker.focal_length_px is not None,
                "detected_object": str(result.label) if result.found and result.mode.value == "scan" else "",
                "confidence": float(result.confidence) if result.found else 0.0,
                "x_offset": float(x_offset),
                "y_offset": float(y_offset),
                "tracking_offset": float(result.tracking_offset),
                "battery": 100,  # TODO: Implement battery monitoring
                "obstacle_detected": False,  # TODO: Implement ultrasonic sensor
                "timestamp": datetime.now().isoformat()
            }

            # Ensure all values are JSON serializable
            status = to_json_serializable(status)
            await websocket.send(json.dumps(status))

        except Exception as e:
            logger.error(f"Error sending status: {e}")
            import traceback
            traceback.print_exc()

    async def broadcast_status(self):
        """Periodically broadcast robot status to all clients"""
        while self.running:
            if self.clients:
                # Create status message
                try:
                    frame, result = self.robot.camera.process_frame()

                    # Get normalized x, y coordinates from the result
                    x_offset, y_offset = 0.0, 0.0
                    if result.mode.value == "follow" and result.found and result.center:
                        # Calculate normalized offset from center (ensure Python float)
                        h, w = frame.shape[:2] if frame is not None else (480, 640)
                        x_offset = float((int(result.center[0]) - w / 2) / (w / 2))
                        y_offset = float((int(result.center[1]) - h / 2) / (h / 2))

                    status = {
                        "type": "status",
                        "tracking": bool(self.robot.tracking_enabled),
                        "emergency_stop": bool(self.robot.emergency_stop),
                        "target_locked": bool(result.found),
                        "distance": float(result.distance) if result.found else 0.0,
                        "mode": str(self.robot.camera.mode.value),
                        "calibrated": self.robot.camera.aruco_tracker.focal_length_px is not None,
                        "detected_object": str(result.label) if result.found and result.mode.value == "scan" else "",
                        "confidence": float(result.confidence) if result.found else 0.0,
                        "x_offset": float(x_offset),
                        "y_offset": float(y_offset),
                        "tracking_offset": float(result.tracking_offset),
                        "battery": 100,
                        "obstacle_detected": False,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Ensure all values are JSON serializable (convert numpy types)
                    status = to_json_serializable(status)

                    # Process vision result for motor control
                    if not self.robot.emergency_stop and self.robot.tracking_enabled:
                        # Log motor control activity every 30 frames
                        if hasattr(self, '_motor_log_counter'):
                            self._motor_log_counter += 1
                        else:
                            self._motor_log_counter = 0

                        if self._motor_log_counter % 30 == 0:
                            logger.info(
                                f"🎮 Motor Control Active: "
                                f"Mode={result.mode.value}, Found={result.found}, "
                                f"Distance={result.distance:.2f}m, Offset={result.tracking_offset:+.3f}"
                            )

                        self.robot.process_vision_result(result)

                    # Broadcast to all clients
                    message = json.dumps(status)
                    disconnected = set()

                    for client in self.clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)

                    # Remove disconnected clients
                    self.clients -= disconnected

                    # Send video frame if streaming enabled
                    if self.stream_video and frame is not None:
                        await self.broadcast_video_frame(frame)

                except Exception as e:
                    logger.error(f"Error in broadcast loop: {e}")
                    import traceback
                    traceback.print_exc()

            await asyncio.sleep(0.1)  # 10Hz update rate

    async def broadcast_video_frame(self, frame):
        """Broadcast video frame to all clients"""
        import cv2
        import base64

        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.video_quality])
            frame_bytes = base64.b64encode(buffer).decode('utf-8')

            message = {
                "type": "video_frame",
                "frame": frame_bytes,
                "timestamp": datetime.now().isoformat()
            }

            # Broadcast to all clients
            disconnected = set()
            message_str = json.dumps(message)

            for client in self.clients:
                try:
                    await client.send(message_str)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)

            # Remove disconnected clients
            self.clients -= disconnected

        except Exception as e:
            logger.error(f"Error broadcasting video frame: {e}")

    async def start(self, host="0.0.0.0", port=8765):
        """Start WebSocket server"""
        self.running = True

        # Start server
        async with websockets.serve(self.handler, host, port):
            logger.info(f"WebSocket server started on {host}:{port}")

            # Start status broadcast loop
            await self.broadcast_status()

    def stop(self):
        """Stop the server"""
        self.running = False
        logger.info("WebSocket server stopped")
