# IoT + Edge AI Voice Command System

## 📌 Overview
This project develops an IoT system that responds to pre-set voice commands and controls smart devices accordingly. The system uses an **ESP32** (or ESP32C3) microcontroller with a lightweight neural network trained to recognize commands. Commands are processed and sent to a **Firebase** database, which updates connected ESP32 devices to control lights and speakers in different rooms.

## 🚀 Features
- 🎤 **Voice Command Recognition**: Uses a **TensorFlow Lite** model trained on the Fluent Speech Commands Voice Dataset to interpret commands.
- ⚡ **Edge AI Processing**: Runs an optimized model on an **ESP32 microcontroller** for real-time inference.
- 📡 **Real-Time Device Control**: Updates **Firebase Realtime Database** to trigger ESP32-controlled devices.
- 🏠 **Multi-Room Support**: Sends commands to devices in specific locations (e.g., kitchen, living room).
- 📟 **Text Display**: Displays executed commands on a **small LED screen**.

## 🛠 Hardware Requirements
- **ESP32** (preferred) or **ESP32C3** microcontroller
- Additional ESP32 boards (for device control)
- **WS2812B LED Strip**
- **DFPlayer Mini MP3 Module + Speaker**
- **Small LED Display**

## 🖥 Software & Tools
- **TensorFlow Lite** for model training & optimization
- **Firebase Realtime Database** for communication
- **Arduino IDE** / **PlatformIO** for ESP32 programming


## ⚙️ Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/iot-edge-ai-voice.git
   cd iot-edge-ai-voice
   ```
2. **Install dependencies** (TensorFlow Lite, Firebase SDK, ESP32 board support).
3. **Flash the ESP32** with the voice recognition firmware.
4. **Connect ESP32-controlled devices** to Firebase and deploy the system.

## 🔮 Future Improvements
- Expand voice command support
- Integrate additional IoT devices
- Improve speech recognition accuracy

---
Made by Keegan Miller
