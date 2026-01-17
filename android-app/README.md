# Grocery Buddy - Android App

Android app for controlling the Grocery Buddy robot.

## Tech Stack

- **Language**: Kotlin
- **UI Framework**: Jetpack Compose
- **Networking**: OkHttp WebSocket
- **Architecture**: MVVM

## Project Structure

```
app/src/main/java/com/grocerybuddy/
├── MainActivity.kt
├── ui/
│   ├── screens/
│   │   ├── HomeScreen.kt
│   │   ├── CalibrationScreen.kt
│   │   └── GroceryListScreen.kt
│   ├── components/
│   │   ├── StatusCard.kt
│   │   ├── GroceryItem.kt
│   │   └── ControlButtons.kt
│   └── theme/
│       └── Theme.kt
├── data/
│   ├── GroceryDatabase.kt
│   ├── GroceryItem.kt
│   └── GroceryDao.kt
├── network/
│   └── RobotWebSocket.kt
└── viewmodel/
    └── RobotViewModel.kt
```

## Setup

1. Open project in Android Studio
2. Minimum SDK: 26 (Android 8.0)
3. Target SDK: 34
4. Sync Gradle dependencies
5. Run on device or emulator

## Features

- Real-time robot status dashboard
- WebSocket communication with Raspberry Pi
- Calibration button for target acquisition
- Start/Stop tracking controls
- Emergency stop button
- Grocery list management

## Configuration

Update the WebSocket URL in `RobotWebSocket.kt`:
```kotlin
private val serverUrl = "ws://raspberrypi.local:8765"
```
