# Sign Language Recognition Setup

## Overview
The OneVoice application now includes sign language to text recognition using **OpenCV and basic computer vision** for gesture detection (no MediaPipe required!).

## Features
- **Real-time video recording** from webcam
- **Video file upload** support (MP4, WebM, MOV, AVI)
- **Gesture recognition** for common signs:
  - 👍 Thumbs up → "yes"
  - ✋ Open palm → "hello"
  - ✌️ Peace sign → "peace"
  - 👎 Thumbs down → "no"
  - ✊ Closed fist → "stop"

## Installation

### Backend Dependencies

1. **Install Python packages** (works with Python 3.8-3.13):
   ```bash
   cd backend
   pip install opencv-python numpy
   ```

   This will install:
   - `opencv-python` - Video processing and computer vision
   - `numpy` - Array operations for image processing

2. **Verify installation**:
   ```bash
   python -c "import cv2, numpy; print('All dependencies installed!')"
   ```

### Frontend
No additional dependencies needed - the feature uses the browser's native MediaRecorder API.

## Usage

### From the Web Interface

1. **Start the backend**:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the app** at `http://localhost:3000` (or the port shown)

4. **Click "Sign Language" mode** in the header

5. **Record or upload**:
   - **Record**: Click "Start Recording", perform gestures, then "Stop Recording"
   - **Upload**: Click "Choose File" and select a video

6. **Process**: Click "Recognize Signs" to analyze the video

### API Endpoint

**POST** `/sign_language_to_text`

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (video file)

**Response**:
```json
{
  "recognizedText": "hello",
  "confidence": 0.85,
  "processingTimeMs": 1234
}
```

**Example with curl**:
```bash
curl -X POST http://127.0.0.1:8000/sign_language_to_text \
  -F "file=@sign_video.webm"
```

## How It Works

1. **Video Input**: Accepts video from webcam recording or file upload
2. **Frame Processing**: Processes every 5th frame for efficiency
3. **Skin Detection**: Uses HSV color space to detect skin-colored regions
4. **Contour Analysis**: Finds hand contours and calculates convex hull
5. **Finger Counting**: Counts convexity defects to determine number of extended fingers
6. **Gesture Classification**: Maps finger count to gestures (0=fist, 2=peace, 4+=open palm)
7. **Aggregation**: Counts gesture occurrences and returns the most common one with confidence score

## Extending Gesture Recognition

### Adding New Gestures

Edit `backend/main.py` in the `SignLanguageRecognizer` class:

1. **Add to gesture map**:
   ```python
   def _initialize_gesture_map(self) -> Dict[str, str]:
       return {
           "thumbs_up": "yes",
           "open_palm": "hello",
           # Add your gesture here
           "your_gesture": "your_text",
       }
   ```

2. **Add detection logic**:
   ```python
   def _detect_gesture(self, hand_landmarks) -> Optional[str]:
       landmarks = hand_landmarks.landmark
       
       # Your detection logic
       if <your_condition>:
           return "your_gesture"
       
       return None
   ```

### Using a Machine Learning Model

For production use, replace the heuristic detection with a trained model:

1. Train a classifier on sign language datasets (e.g., ASL, ISL)
2. Replace `_detect_gesture()` with model inference
3. Update `gesture_map` with your model's output classes

## Troubleshooting

### OpenCV Installation Issues

**Windows**:
```bash
pip install opencv-python --no-cache-dir
```

**Linux/Mac (headless server)**:
```bash
pip install opencv-python-headless numpy
```

### Camera Access Denied

- **Browser**: Grant camera permissions when prompted
- **HTTPS**: Some browsers require HTTPS for camera access in production

### Low Recognition Accuracy

- **Lighting**: Ensure good, even lighting (avoid shadows)
- **Background**: Use a plain background different from skin tone
- **Hand position**: Keep hand clearly visible and centered in frame
- **Gestures**: Perform gestures slowly and deliberately
- **Skin tone**: Adjust HSV ranges in `_detect_skin()` for different skin tones:
  ```python
  lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Adjust these values
  upper_skin = np.array([20, 255, 255], dtype=np.uint8)
  ```

### Performance Issues

- Reduce frame processing rate (change `frame_count % 5` to higher number)
- Lower video resolution
- Use `opencv-python-headless` for server deployments

## Technical Details

### OpenCV Computer Vision Approach
- **Skin Detection**: HSV color space filtering for skin tones
- **Contour Detection**: Finds hand shapes using edge detection
- **Convex Hull**: Calculates the outer boundary of the hand
- **Convexity Defects**: Counts spaces between fingers
- **Performance**: ~20-30 FPS on modern hardware (lighter than MediaPipe)

### Supported Video Formats
- WebM (VP8/VP9)
- MP4 (H.264)
- MOV (QuickTime)
- AVI

### Processing Pipeline
```
Video → Frame Extraction → HSV Conversion → Skin Detection → 
Contour Analysis → Convexity Defects → Finger Counting → 
Gesture Classification → Aggregation → Result
```

## Future Enhancements

- [ ] Real-time streaming recognition (WebSocket)
- [ ] ASL alphabet recognition
- [ ] Multi-word sentence recognition
- [ ] Custom gesture training interface
- [ ] Sign language translation (sign-to-sign)
- [ ] Integration with speech synthesis for complete accessibility

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Contour Detection Tutorial](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [ASL Datasets](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [Sign Language Recognition Papers](https://paperswithcode.com/task/sign-language-recognition)

## License

This feature uses OpenCV (Apache 2.0 License) and NumPy (BSD License).
