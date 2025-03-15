# Security Camera Application

## Overview
This application turns your webcam into a smart security camera with motion detection capabilities. Built with Python and OpenCV, it offers both manual recording and automated sentry mode for surveillance.

## Features

### 1. Live Video Feed
- Real-time display from your webcam
- Professional camera frame overlay for better visual feedback
- FPS optimization for smooth video rendering

### 2. Manual Recording
- Start/stop recording with a simple spacebar press
- Videos saved with timestamp filenames
- AVI format with XVID codec for good quality and reasonable file size

### 3. Sentry Mode
- Automated motion detection and recording
- Intelligent background modeling to adapt to environmental changes
- Only records when motion is detected to save storage space
- Visual indicators when motion is detected

### 4. User Interface
- Clear status indicators for recording and sentry mode status
- Background model visualization for debugging
- Countdown timer when initializing sentry mode

## How to Use

### Prerequisites
- Python 3.x
- OpenCV (cv2)
- NumPy

### Installation
```bash
pip install numpy opencv-python
```

### Controls
- **ESC**: Exit the application
- **Spacebar**: Start/stop manual recording (when not in sentry mode)
- **S**: Toggle sentry mode on/off

### Sentry Mode Setup
When activating sentry mode:
1. A 5-second countdown begins, allowing you to clear any moving objects from view
2. The system collects 1000 background frames to build a reliable motion detection model
3. Once initialized, the system automatically begins recording when motion is detected
4. Recording stops when motion ceases

## Configuration
The following parameters can be adjusted at the top of the script:

```python
video_name = "Laptop Webcam"        # Display name for the video feed
video_source_path = 0               # Camera index (0 for default webcam)
video_save_format = "avi"           # Output video format
video_save_format_fourcc = "XVID"   # Video codec

# Motion detection parameters
blur_ksize = (9, 9)                 # Blur kernel size
blur_sigma = 3                      # Blur sigma value
diff_threshold = 150                # Difference threshold for motion detection
bg_update_rate = 0.05               # Background update rate
fg_update_rate = 0.001              # Foreground update rate
```

## How It Works

### Motion Detection Algorithm
1. The application builds a background model by averaging multiple frames
2. For each new frame:
   - Apply Gaussian blur to reduce noise
   - Calculate the difference between the current frame and background model
   - Threshold the difference to identify potential motion
   - Apply erosion and dilation to reduce false positives
   - Update the background model adaptively
   - Determine if significant motion is present

### Recording Management
- In manual mode, recording starts and stops with user input
- In sentry mode, recording automatically starts when motion is detected and stops when motion ceases
- All videos are saved with timestamped filenames

## Limitations and Future Improvements
- Currently supports only webcam input; could be extended to support IP cameras or video files
- Motion detection parameters may need adjustment for different environments
- No cloud storage or remote notification features yet

## License
The code is licensed under the GNU AGPL v3.0.

For more details, see the `LICENSE` file under each directory or visit [GNU AGPL v3.0 License](https://www.gnu.org/licenses/agpl-3.0.html)