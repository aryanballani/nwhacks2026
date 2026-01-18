# GlideCart 🛒🤖

**GlideCart** is an autonomous, AI-powered shopping companion designed to remove physical barriers in retail. Built in just 12 hours, this prototype provides a hands-free, intelligent shopping experience for people with mobility challenges or accessibility needs.

### Video Demo:
https://youtube.com/shorts/Cf38NwVQo1U

## 🚀 The Vision

For wheelchair users or the elderly, navigating a store while pushing a heavy cart is a significant physical hurdle. **GlideCart** solves this by using vision-based tracking to follow the user and integrated computer vision to automate the checkout process.

## ✨ Key Features

* **"Follow-Me" Mode:** Uses **ArUco marker** calibration to lock onto the user and maintain a safe distance autonomously.
* **Smart Recognition:** A custom-trained, lightweight vision model identifies items (Milk, Apples, Oranges, etc.) in real-time.
* **Live Checklist:** Automatically detects and checks off items from a shopping list as they are placed in the cart.
* **Mobile Control:** An **Android App** provides a live camera feed and manual override options.
* **Real-time Sync:** Powered by **Supabase SQL** for instant communication between the robot and the mobile device.

## 🛠️ Tech Stack

* **Hardware:** Raspberry Pi 4, USB Webcam, L298N Motor Driver, DC Motors.
* **Vision:** OpenCV (ArUco tracking), Python, TensorFlow Lite (Custom Object Detection).
* **Mobile/Backend:** Android SDK (Kotlin/Java), Gradle, Supabase (PostgreSQL).
* **Connectivity:** Internet Gateway for Pi-to-App low-latency communication.

## 🏗️ Architecture

1. **Perception:** The Webcam captures frames; OpenCV detects the ArUco marker for distance/angle.
2. **Logic:** The Raspberry Pi calculates motor speeds to maintain a "lock" on the user.
3. **Recognition:** The Vision model scans the cart's interior; detected items are sent to the backend.
4. **Interface:** Supabase pushes the update to the Android App, checking the item off the user's list.

## 📈 Future Scalability

This MVP proves the concept on lightweight hardware. With more powerful processors (e.g., NVIDIA Jetson) and depth sensors (LiDAR), GlideCart can scale to handle:

* Dynamic obstacle avoidance in high-traffic aisles.
* Full store navigation and inventory mapping.
* Integration with hospital and warehouse logistics.

## 👥 The Team

Built with 💡 and ☕ in 12 hours for the NwHacks 2026.

