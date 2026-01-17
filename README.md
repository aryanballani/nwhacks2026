# Grocery Buddy 🛒

Autonomous Shopping Cart Robot with Person-Following & Android Control

## Overview

A smart robot cart that autonomously follows a user (identified by a colored marker) through a grocery store, while an Android app provides control, calibration, and grocery list management.

## Key Features

- **Accessibility Focus** - Helps people with mobility challenges
- **Computer Vision** - Real-time person tracking with OpenCV
- **Mobile Integration** - Seamless Android control via WebSocket
- **Practical Application** - Solves a real problem

## System Components

- **Android App** (Kotlin + Jetpack Compose) - User control interface
- **Raspberry Pi** (Python) - Computer vision, motor control, WebSocket server
- **Hardware** - Pi Camera, DC motors, L298N driver, ultrasonic sensor

## Repository Structure

```
grocery-buddy/
├── android-app/          # Android Kotlin app
├── raspberry-pi/         # Raspberry Pi Python code
├── docs/                 # Documentation and assembly guides
└── README.md
```

## Quick Start

See documentation in `/docs` for:
- Assembly instructions
- Software setup
- Wiring diagrams
- Demo guide

## Tech Stack

### Android App
- Kotlin
- Jetpack Compose
- OkHttp (WebSocket)
- Material 3

### Raspberry Pi
- Python 3
- OpenCV
- pigpio (motor control)
- websockets

## License

MIT
