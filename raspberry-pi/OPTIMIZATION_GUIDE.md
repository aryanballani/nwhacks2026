# Raspberry Pi Performance Optimization Guide

## Issues Fixed

### 1. JSON Serialization Error
**Problem:** `Object of type float32 is not JSON serializable`
- NumPy float32 values were being passed to `json.dumps()` without conversion

**Solution:** Convert all numeric values to Python float using `float()` before JSON serialization
- Files modified: `server/websocket_server.py`

### 2. Performance Bottlenecks
**Problems identified:**
- Synchronous processing in websocket broadcast loop (10Hz)
- Camera capture blocking frame processing
- YOLO/ONNX inference happening in main thread
- High resolution (640x480) and FPS (30) consuming too much CPU
- Video streaming with high JPEG quality (50%)

**Solutions implemented:**

#### A. Multithreading Architecture
- **Capture Thread**: Continuously captures frames from camera
- **Processing Thread**: Processes frames (ArUco/YOLO detection)
- **Main Thread**: Handles websockets, motor control, and frame annotation
- Uses thread-safe queues with size limits to prevent lag

#### B. Camera Configuration Optimizations
```python
CAMERA_WIDTH = 416   # Reduced from 640
CAMERA_HEIGHT = 416  # Reduced from 480 (square for YOLO efficiency)
CAMERA_FPS = 15      # Reduced from 30
```

#### C. Frame Skipping
- SCAN mode: Process every 3rd frame (skip_frames_scan = 2)
- FOLLOW mode: Process every frame (ArUco is fast)
- Cache last result for skipped frames

#### D. Video Streaming Optimization
- Reduced JPEG quality from 50% to 30%
- Smaller frames (416x416) reduce encoding time

#### E. Comprehensive Logging
Added detailed timing logs to identify bottlenecks:
```
[FOLLOW] Capture: 15.2ms | Process: 8.3ms | Annotate: 12.1ms | Total: 35.6ms
```

## Architecture Changes

### Before (Synchronous)
```
Websocket Broadcast Loop (10Hz):
  1. Capture frame ----\
  2. Process frame      |---> All blocking, sequential
  3. Annotate frame     |
  4. Send to clients ---/
```

### After (Multithreaded)
```
Thread 1 (Capture):     Thread 2 (Processing):    Main (Websocket):
  Capture frame ------>   Process frame ------->   Get result
       |                       |                   Annotate
       |                       |                   Send to clients
  Capture frame ------>   Process frame ------->   Get result
```

## Performance Improvements Expected

### CPU Load
- Before: ~80-100% CPU usage on single core
- After: Distributed across multiple cores (~50-60% per core)

### Latency
- Before: 100-150ms per frame (sequential)
- After: 30-50ms effective latency (parallel processing)

### FPS
- Before: ~7-10 FPS in FOLLOW mode with lag
- After: ~15 FPS smooth operation

## Configuration Options

### Enable/Disable Threading
```python
# In main_server.py or when initializing CameraController
camera = CameraController(camera_id=0, use_yolo=True, threaded=True)  # Enable threading
camera = CameraController(camera_id=0, use_yolo=True, threaded=False) # Disable threading
```

### Adjust Frame Skipping
```python
# In vision/camera_controller.py __init__
self._skip_frames_scan = 2    # Process every 3rd frame in SCAN mode
self._skip_frames_follow = 0  # Process every frame in FOLLOW mode
```

### Camera Resolution Tuning
```python
# In vision/config.py
CAMERA_WIDTH = 416   # Lower = faster, but less accurate
CAMERA_HEIGHT = 416
CAMERA_FPS = 15      # Lower = less CPU load
```

## Debugging and Monitoring

### Enable Debug Logging
```python
# Add to top of your script
import logging
logging.basicConfig(level=logging.DEBUG)  # Change from INFO to DEBUG
```

### Monitor System Resources on Raspberry Pi
```bash
# CPU and memory usage
htop

# CPU temperature (important for throttling)
vcgencmd measure_temp

# Check if CPU is throttled
vcgencmd get_throttled
```

### Performance Logs
The system now logs detailed timing information every 30 frames:
- Capture time: How long it takes to read from camera
- Process time: ArUco detection or YOLO inference time
- Annotate time: OpenCV drawing operations
- Total time: Sum of all steps

## Troubleshooting

### Camera feed still slow
1. **Check CPU temperature**: If > 80°C, Pi may be throttling
   ```bash
   watch -n 1 vcgencmd measure_temp
   ```

2. **Reduce resolution further**:
   ```python
   CAMERA_WIDTH = 320
   CAMERA_HEIGHT = 320
   ```

3. **Increase frame skipping**:
   ```python
   self._skip_frames_scan = 4  # Process every 5th frame
   ```

4. **Disable video streaming** in the app (only send status updates)

### App still lags
1. **Increase websocket broadcast interval**:
   ```python
   # In server/websocket_server.py, line ~212
   await asyncio.sleep(0.2)  # Changed from 0.1 (reduce from 10Hz to 5Hz)
   ```

2. **Reduce annotation** (comment out heavy OpenCV drawing in camera_controller.py)

### ArUco not detecting properly
1. **Disable frame skipping in FOLLOW mode**:
   ```python
   self._skip_frames_follow = 0  # Already set to 0
   ```

2. **Increase marker size** or print at higher resolution

3. **Improve lighting** conditions

### Storage Issues
Check available disk space:
```bash
df -h
```

If low on space:
```bash
# Remove old logs
sudo journalctl --vacuum-time=7d

# Remove unused packages
sudo apt autoremove
sudo apt clean
```

## Additional Optimizations (Optional)

### 1. Use lighter YOLO model
```python
# In vision/config.py
YOLO_MODEL = "models/yolov8n.pt"  # Already using nano model (lightest)
```

### 2. Compile OpenCV with optimizations
```bash
# Rebuild OpenCV with NEON optimizations for ARM
# This is advanced - only if you're comfortable compiling from source
```

### 3. Overclock Raspberry Pi (risky)
```bash
# Edit /boot/config.txt
sudo nano /boot/config.txt

# Add (at your own risk):
over_voltage=6
arm_freq=2000
```

### 4. Run camera processing on separate computer
Instead of running everything on Pi:
- **Laptop**: Runs camera processing (vision/)
- **Raspberry Pi**: Only runs motor control and receives commands

This architecture is already partially implemented in your code (receiver_motor.py).

## Testing Checklist

After deploying optimizations, test:
- [ ] Camera feed is smooth at ~15 FPS
- [ ] ArUco marker detection works in FOLLOW mode
- [ ] Object detection works in SCAN mode
- [ ] Motor control responds promptly to vision commands
- [ ] No JSON serialization errors in logs
- [ ] CPU usage is distributed across cores
- [ ] Temperature stays below 80°C
- [ ] App responds without lag

## Monitoring During Operation

```bash
# SSH into Raspberry Pi
ssh ishman@10.19.129.238

# Run system monitor
htop

# In another terminal, monitor logs
journalctl -u your-service-name -f

# Or if running directly
python3 main_server.py  # Watch the timing logs
```

## Reverting Changes

If issues arise, you can revert to original settings:

```python
# vision/config.py
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# vision/camera_controller.py
threaded=False  # In CameraController initialization
```
