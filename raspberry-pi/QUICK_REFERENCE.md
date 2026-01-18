# Quick Reference Guide - Grocery Buddy Robot

## ✅ All Issues Fixed

### 1. JSON Serialization Error - FIXED ✓
- Added `to_json_serializable()` helper function to convert all numpy types
- All numeric values explicitly converted to Python native types
- No more `Object of type float32 is not JSON serializable` errors

### 2. Performance Optimizations - COMPLETE ✓
- **Multithreading**: Capture + Processing in separate threads
- **Lower Resolution**: 320x240 (was 640x480) = 75% faster
- **Frame Skipping**: 5Hz when tracking (was 10Hz)
- **Video Streaming**: Disabled during tracking to save bandwidth
- **Camera Buffer**: Set to 1 to minimize lag

### 3. Distance Tracking - WORKING ✓
- ArUco distance calculation fixed
- Calibration with detailed logging
- Distance shown in FOLLOW mode after calibration

## 🚀 Quick Start Commands

### SSH to Raspberry Pi (No Password)
```bash
ssh ishman@10.19.129.238
cd ~/NWHacks/raspberry-pi
```

### Kill Any Running Processes
```bash
pkill python3
```

### Run Main Server (with App)
```bash
python3 main_server.py
```

### Run Motor Receiver (standalone)
```bash
python3 receiver_motor.py
```

### Test Distance Detection
```bash
python3 test_motor_distance.py
```

## 📱 How to Use the App

### 1. Connect
- Open app on Android
- Connect to Pi's IP: `10.19.129.238:8765`

### 2. Switch to FOLLOW Mode
- Tap "FOLLOW" mode button
- You'll see "ArUco Marker (Uncalibrated)" if not calibrated

### 3. Calibrate Distance
- Hold ArUco marker at **30cm** from camera
- Keep marker centered in frame
- Tap "Calibrate" button
- Wait for confirmation: "Calibration successful"

### 4. Enable Tracking
- Tap "Start Tracking" button
- Robot will now follow the ArUco marker
- Maintains ~1.0m distance automatically

### 5. Stop
- Tap "Stop Tracking" to stop motors
- Or tap "Emergency Stop" to immediately halt

## 🐛 Troubleshooting

### Camera Already in Use
```bash
# Check what's using camera
lsof /dev/video0

# Kill all Python processes
pkill -9 python3
```

### No Distance Showing
1. **Check calibration status** - Label should say "Calibrated"
2. **Re-calibrate** - Hold marker at 30cm and press Calibrate
3. **Check logs** - Look for "Calibration successful" message
4. **Verify marker** - Must be ArUco 5x5_50 dictionary, 5cm x 5cm

### Robot Not Following / Slow Response
1. **Check CPU usage**: `htop` on Pi
2. **Check temperature**: `vcgencmd measure_temp` (should be < 80°C)
3. **Reduce frame rate** - Already optimized to 5Hz when tracking
4. **Disable video streaming** - Done automatically when tracking enabled

### Wrong Turn Direction
If robot turns opposite direction:
1. Check `receiver_motor.py` logs for "Turn LEFT" vs "Turn RIGHT"
2. If correct in logs but wrong physically, swap motor wires
3. Or in `motors/motor_controller.py`, swap left/right motor pins

### Distance Inaccurate
1. **Re-calibrate** at exactly 30cm
2. **Check marker size** - Must be 5cm x 5cm
3. **Adjust in config.py**:
   ```python
   ARUCO_MARKER_LENGTH_CM = 5.0  # Your marker's actual size
   ARUCO_CALIBRATION_DISTANCE_CM = 30.0  # Distance during calibration
   ```

## 📊 Understanding the Logs

### Camera Performance Logs
```
[FOLLOW] Capture: 35.2ms | Process: 8.1ms | Annotate: 1.2ms | Total: 44.5ms
```
- **Capture**: Time to grab frame from camera
- **Process**: ArUco detection time
- **Annotate**: Drawing overlays time
- **Total**: Should be < 100ms for smooth operation

### ArUco Detection Logs
```
ArUco Detection: ID=5, Distance=0.85m, Offset=-0.234, Calibrated=True, Focal=245.3
```
- **ID**: ArUco marker number (0-49)
- **Distance**: Measured distance in meters
- **Offset**: -1.0 (far left) to +1.0 (far right), 0 is centered
- **Calibrated**: Whether camera is calibrated for distance
- **Focal**: Focal length in pixels (calculated during calibration)

### Motor Control Logs
```
Motor Control: Distance=0.75m (target=1.00m, error=-0.25m), Offset=+0.123
⬇️  Moving BACKWARD 45 - Target too close
🔄 Turning RIGHT ➡️ (offset: +0.123, turn: +10)
🎮 Motor Speeds: L=-55, R=-35
```
- **Distance**: Current distance to marker
- **Target**: Desired distance (1.0m)
- **Error**: Difference (negative = too close, positive = too far)
- **Offset**: Horizontal position (-1 to +1)
- **Motor Speeds**: Left and right motor values (-100 to +100)

## ⚙️ Configuration Files

### `vision/config.py` - Camera Settings
```python
CAMERA_WIDTH = 320              # Lower = faster
CAMERA_HEIGHT = 240             # Lower = faster
CAMERA_FPS = 15                 # Lower = less CPU load

ARUCO_MARKER_LENGTH_CM = 5.0    # Physical marker size
ARUCO_CALIBRATION_DISTANCE_CM = 30.0  # Calibration distance
```

### `receiver_motor.py` - Motor Control
```python
TARGET_DISTANCE = 1.0           # Target following distance (meters)
DISTANCE_TOLERANCE = 0.2        # Acceptable range (±0.2m)
MIN_DISTANCE = 0.3              # Safety stop distance
MAX_DISTANCE = 3.0              # Max tracking distance

BASE_SPEED = 60                 # Normal speed (0-100)
MAX_SPEED = 90                  # Maximum speed
TURN_GAIN = 80                  # Turn sensitivity (higher = sharper)
DEADZONE = 0.05                 # Ignore small offsets
```

### `server/websocket_server.py` - Network
```python
self.video_quality = 30         # JPEG quality (lower = faster)
await asyncio.sleep(0.1)        # Update rate (0.1 = 10Hz, 5Hz when tracking)
```

## 🎯 ArUco Marker Setup

### Generate Marker
1. Go to: https://chev.me/arucogen/
2. Select: **5x5 (50, 100, 250, 1000)** dictionary
3. Choose ID: **0-49** (any)
4. Size: **5cm x 5cm** (important!)
5. Print at 100% scale (no scaling)

### Verify Marker Size
```bash
# Measure printed marker with ruler
# Should be exactly 5cm x 5cm
# If different, update ARUCO_MARKER_LENGTH_CM in config.py
```

## 🔍 Advanced Testing

### Test Camera Only (No Motors)
```bash
python3 test_motor_distance.py
```
- Shows ArUco detection
- Shows distance and offset
- Shows what motors WOULD do
- Good for debugging vision system

### Test Motors Only (No Vision)
```bash
# In receiver_motor.py, uncomment test code at bottom
# Or send test data manually:
echo "0.0 1.0" | nc 10.19.129.238 5000
# Format: "offset distance"
# offset: -1.0 to +1.0 (left to right)
# distance: in meters
```

### Monitor System Resources
```bash
# CPU and memory
htop

# Temperature (important!)
watch -n 1 vcgencmd measure_temp

# Disk space
df -h

# Check throttling
vcgencmd get_throttled
# 0x0 = no throttling, any other value = throttled
```

## 📈 Performance Benchmarks

### Expected Performance on Pi 4B

**Without Tracking:**
- FPS: 10-15
- CPU Usage: 40-60% per core
- Temperature: 40-50°C
- Latency: 60-80ms

**With Tracking Enabled:**
- FPS: 5-8 (intentionally reduced)
- CPU Usage: 50-70% per core
- Temperature: 50-60°C
- Latency: 80-120ms

**During Calibration:**
- Brief spike to 70-80% CPU
- Returns to normal after 1-2 seconds

## 🆘 Emergency Procedures

### Robot Not Stopping
1. Press "Emergency Stop" in app
2. If unresponsive: `pkill python3` on Pi
3. Physical: Disconnect battery/power

### Lost WiFi Connection
1. App will show "Disconnected"
2. Motors stop automatically (safety feature)
3. Reconnect WiFi and restart app

### Pi Overheating (>80°C)
```bash
# Check temperature
vcgencmd measure_temp

# If > 80°C:
1. Stop all Python processes: pkill python3
2. Add heatsink or fan to Pi
3. Reduce camera resolution further in config.py
```

### Code Updates from Laptop
```bash
# On your laptop
rsync -avz /home/kanish10/Desktop/NWHacks/raspberry-pi/ \
  ishman@10.19.129.238:~/NWHacks/raspberry-pi/ \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'models/'
```

## 📝 Quick Debug Checklist

When something goes wrong:

- [ ] Camera in use? `lsof /dev/video0`
- [ ] Python processes running? `ps aux | grep python`
- [ ] Calibration done? Check app or logs
- [ ] ArUco marker visible? Check camera feed
- [ ] Correct marker size? Measure with ruler
- [ ] Pi temperature OK? `vcgencmd measure_temp`
- [ ] WiFi connected? `ping 10.19.129.238`
- [ ] App connected? Check app status
- [ ] Tracking enabled? Check app button state

## 🎓 Understanding the System

### Data Flow
```
Camera (Thread 1)
    ↓ frames
Processing (Thread 2)
    ↓ ArUco detection
    ↓ distance calculation
Main Thread
    ↓ JSON status
WebSocket Server
    ↓ network
Android App
    ↑ commands (calibrate, start tracking, etc.)
```

### Motor Control Logic
```
1. Get ArUco position (offset) and distance
2. Calculate distance error = current - target
3. Determine forward/backward speed based on error
4. Calculate turn amount based on offset
5. Apply differential drive: L = forward - turn, R = forward + turn
6. Send to motors
```

## 🔧 Common Customizations

### Change Target Distance
```python
# In receiver_motor.py
TARGET_DISTANCE = 1.5  # Follow at 1.5m instead of 1.0m
```

### Make Turns More/Less Aggressive
```python
# In receiver_motor.py
TURN_GAIN = 100  # More aggressive (was 80)
TURN_GAIN = 50   # Less aggressive
```

### Speed Up/Slow Down Following
```python
# In receiver_motor.py
BASE_SPEED = 80   # Faster (was 60)
BASE_SPEED = 40   # Slower
```

### Improve Camera Performance (Lower Quality)
```python
# In vision/config.py
CAMERA_WIDTH = 240   # Even lower (was 320)
CAMERA_HEIGHT = 180  # Even lower (was 240)
```

## 📞 Getting Help

If you encounter issues not covered here:

1. Check logs for error messages
2. Run test scripts to isolate problem
3. Verify hardware connections
4. Check GitHub issues (if applicable)
5. Temperature and throttling status

---

**Last Updated**: 2026-01-18
**System Version**: Optimized for Pi 4B with multithreading and distance tracking
