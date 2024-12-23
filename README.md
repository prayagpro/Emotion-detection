**Emotion Detection System**: (Level- Medium)

```markdown
# Emotion Detection System

An AI-powered real-time **Emotion Detection System** that detects faces and recognizes emotions from a webcam feed. The system uses a pre-trained deep learning model to classify emotions such as happy, sad, angry, surprised, and more.

---

## 📸 Features

- Real-time **Face Detection** using OpenCV's Haar Cascade.
- Emotion recognition powered by a pre-trained `_mini_XCEPTION` model.
- Displays probabilities of detected emotions with visual feedback.
- Compatible with live webcam input.

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotion-detection-system.git
cd emotion-detection-system
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
Install the required libraries:
```bash
pip install tensorflow keras numpy opencv-python imutils
```

## 🚀 Running the Project

Start the Emotion Detection System:
```bash
python face_detection.py
```

### Key Features
1. **Live Webcam Feed:**
   - Detects faces and displays bounding boxes.
   - Predicts and shows emotions with probabilities.
2. **Exit:**
   - Press `q` to quit the program.

## 📦 Dependencies

The following Python libraries are required:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Imutils

Install them using:
```bash
pip install tensorflow keras numpy opencv-python imutils
```


## 🛠️ How It Works

1. **Face Detection:**  
   - The Haar Cascade (`haarcascade_frontalface_default.xml`) detects faces in video frames.  

2. **Emotion Prediction:**  
   - The `_mini_XCEPTION.102-0.66.hdf5` model predicts emotions for detected faces.  

3. **Display:**  
   - Bounding boxes and emotion labels are displayed on the webcam feed.  


## 🎯 Future Enhancements

- Support for detecting and analyzing multiple faces simultaneously.
- Add additional emotions like confusion, excitement, etc.
- GPU acceleration for faster performance.
- Save video output with detected emotions and annotations.

## 🧑‍💻 Contributing

Contributions are welcome!  
Feel free to open an issue or submit a pull request.

## 🙌 Acknowledgments

- **[OpenCV](https://opencv.org/):** For face detection.  
- **Mini XCEPTION Model:** For emotion detection.
