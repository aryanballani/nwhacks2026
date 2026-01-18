#!/bin/bash

# Start Grocery Buddy WebSocket Server
# Run with: sudo ./start_server.sh

cd ~/Desktop/NWHacks/raspberry-pi

echo "========================================"
echo "Starting Grocery Buddy Server"
echo "========================================"
echo "Camera: video1"
echo "YOLO: Enabled"
echo "Port: 8765"
echo ""

sudo python3 main_server.py
