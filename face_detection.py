from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import cv2
import numpy as np
import imutils

# Paths to models and files
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'

# Load Haar Cascade for face detection
face_detection = cv2.CascadeClassifier(detection_model_path)

# Load pre-trained emotion detection model
emotion_classifier = load_model(emotion_model_path, compile=False)

# Emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Function to process and predict emotions
def detect_and_predict_emotions(frame):
    # Convert the frame to grayscale (Haar Cascade works on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detection.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a copy of the frame for drawing bounding boxes
    frame_clone = frame.copy()

    if len(faces) > 0:
        for (fX, fY, fW, fH) in faces:
            # Extract ROI (Region of Interest) for the detected face
            roi = gray_frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))  # Resize to model input size
            roi = roi.astype("float32") / 255.0  # Normalize pixel values
            roi = img_to_array(roi)  # Convert to array
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Predict the emotion
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # Draw bounding box and label around the face
            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 2)
            cv2.putText(frame_clone, f"{label}: {emotion_probability:.2f}", (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    return frame_clone, faces, preds if len(faces) > 0 else None

# Main function to run the webcam feed and process frames
def main():
    # Start video capture from the webcam
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Emotion Detection')

    while True:
        # Capture frame from the webcam
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to capture video. Exiting...")
            break

        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=400)

        # Detect faces and predict emotions
        processed_frame, faces, preds = detect_and_predict_emotions(frame)

        # Display the processed frame
        cv2.imshow('Emotion Detection', processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()