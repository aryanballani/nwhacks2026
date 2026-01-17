# Grocery Buddy - Quick Start Guide

## 🚀 Get Running in 3 Steps

### 1. Install Dependencies

**Ubuntu/Linux:**
```bash
sudo apt install -y python3-opencv python3-numpy
```

**Optional (for YOLO object detection):**
```bash
pip3 install ultralytics --user
```

### 2. Test Vision System

```bash
cd raspberry-pi
python3 test_vision.py
```

**Controls:**
- `M` - Toggle mode (FOLLOW ↔ SCAN)
- `C` - Calibrate person marker (in FOLLOW mode)
- `Q` - Quit
- `SPACE` - Screenshot

### 3. Run Full Robot

```bash
python3 main.py
```

**Controls:**
- `T` - Toggle tracking ON/OFF (default: OFF)
- `M` - Toggle mode (FOLLOW ↔ SCAN)
- `C` - Calibrate person marker
- `E` - Emergency stop
- `Q` - Quit

## 📋 Modes

### FOLLOW Mode
- Tracks person with colored marker
- **Setup:** Hold pink/magenta object, press `C` to calibrate
- Press `T` to enable tracking
- Robot maintains ~1m distance

### SCAN Mode
- Detects grocery items (banana, apple, orange)
- Uses YOLO (if installed) or color detection
- No calibration needed
- Just show objects to camera

## 🎨 Color Markers

For best tracking in FOLLOW mode:
- **Pink/Magenta** (recommended)
- **Neon Green**
- **Neon Orange**

Minimum size: 15cm x 15cm

## 💡 Tips

1. **Laptop Testing:** Motors are mocked - safe to test anywhere
2. **Good Lighting:** Helps both person tracking and object detection
3. **Tracking Safety:** Tracking is OFF by default, press `T` to enable
4. **Distance:** Keep 0.5-2m from camera for best results

## 🔧 Troubleshooting

**No camera found:**
```bash
ls /dev/video*  # Check camera devices
python3 -c "import cv2; print(cv2.__version__)"  # Check OpenCV
```

**YOLO not available:**
- That's OK! System falls back to color detection automatically
- Install with: `pip3 install ultralytics --user`

**Motors not working on Pi:**
```bash
sudo pigpiod  # Start GPIO daemon
```

## 📖 Full Documentation

See [README.md](README.md) for complete documentation.
