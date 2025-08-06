# AR Pong 🏓

A computer vision-powered augmented reality pong game that overlays a virtual game on real-world surfaces using hand tracking and chessboard detection.

![AR Pong Demo](https://img.shields.io/badge/OpenCV-4.x-blue) ![Python](https://img.shields.io/badge/Python-3.7+-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)

## 🎯 Project Overview

AR Pong combines computer vision, augmented reality, and real-time hand tracking to create an immersive gaming experience. Players can play pong on any flat surface by simply placing a chessboard pattern on it and using hand gestures to control the paddle.

### Key Features

- **📷 Real-time Computer Vision**: Uses OpenCV for chessboard detection and pose estimation
- **🖐️ Hand Tracking**: MediaPipe integration for gesture-based paddle control  
- **🎮 Augmented Reality**: Virtual game elements overlaid on real-world surfaces
- **⚙️ Camera Calibration**: Precise camera calibration for accurate AR projection
- **🎯 Progressive Difficulty**: Ball speed increases with score for enhanced gameplay

## 🛠️ Technical Implementation

### Core Technologies
- **OpenCV**: Computer vision operations, chessboard detection, pose estimation
- **MediaPipe**: Hand landmark detection and tracking
- **NumPy**: Mathematical operations and transformations
- **Python**: Main programming language

### Computer Vision Pipeline

1. **Camera Calibration**: 
   - Automated calibration using chessboard patterns
   - Calculates camera intrinsic parameters and distortion coefficients
   - Stores calibration data for consistent AR accuracy

2. **Surface Detection**:
   - Real-time chessboard corner detection
   - 3D pose estimation using `solvePnP`
   - Perspective transformation for normalized coordinates

3. **Hand Tracking**:
   - MediaPipe hand landmark detection
   - Wrist position mapping to game coordinates
   - Real-time paddle control with collision boundaries

4. **Game Logic**:
   - Physics simulation with collision detection
   - Progressive difficulty scaling
   - Score tracking and game state management

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy mediapipe
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/AR-Pong.git
cd AR-Pong
```

2. Print a chessboard pattern (9x6 corners, 25mm squares)
3. Run camera calibration:
```bash
python calibrate_camera.py
```

4. Start the AR Pong game:
```bash
python ar_hockey_pong_chessboard.py
```

### Usage

1. **Calibration Phase**:
   - Show the chessboard to your camera from various angles
   - Press `SPACE` to capture calibration images (minimum 15 required)
   - Press `Q` when you have enough images to compute calibration

2. **Game Phase**:
   - Place the chessboard on a flat surface
   - Position your hand above the virtual game area
   - Move your hand left/right to control the paddle
   - Keep the ball in play to increase your score!

## 📁 Project Structure

```
AR-Pong/
├── ar_hockey_pong_chessboard.py  # Main game application
├── calibrate_camera.py           # Camera calibration utility
├── calibration_data.npz          # Stored calibration parameters
└── README.md                     # Project documentation
```

## 🎮 Game Mechanics

- **Paddle Control**: Hand position controls paddle movement
- **Ball Physics**: Realistic collision detection with walls and paddle
- **Scoring System**: Points awarded for successful bounces off the top wall
- **Progressive Difficulty**: Ball speed increases with score
- **Visual Feedback**: Real-time hand tracking visualization

## 🔧 Configuration

Key parameters can be modified in the source code:

```python
# Chessboard specifications
CHESSBOARD_SIZE = (9, 6)      # Inner corners
SQUARE_SIZE_METERS = 0.025    # 25mm squares

# Game area dimensions  
TABLE_WIDTH_METERS = 0.4      # 40cm width
TABLE_HEIGHT_METERS = 0.6     # 60cm height

# Camera settings
CAMERA_INDEX = 1              # Camera device index
```

## 🎯 Technical Highlights

- **Real-time Performance**: Optimized computer vision pipeline for smooth gameplay
- **Robust Tracking**: Handles lighting changes and partial occlusions
- **Coordinate Transformations**: Seamless mapping between 3D world coordinates and 2D image space
- **Error Handling**: Graceful fallbacks for calibration and detection failures
- **Modular Design**: Separate calibration and game modules for maintainability

## 🚀 Future Enhancements

- [ ] Multi-player support with dual hand tracking
- [ ] Power-ups and special effects
- [ ] Different game modes (hockey, breakout variants)
- [ ] Mobile app integration
- [ ] Machine learning for improved hand gesture recognition

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest improvements
- Add new game modes or features  
- Optimize performance or add platform support
- Improve documentation

## 👨‍💻 About

This project demonstrates practical applications of:
- Computer vision and augmented reality
- Real-time image processing
- 3D geometry and coordinate transformations
- Interactive user interface design
- Game development with Python

Built as a portfolio project showcasing expertise in computer vision, Python development, and creative problem-solving.

---
