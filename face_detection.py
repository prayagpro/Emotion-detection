from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import cv2
import numpy as np
import imutils

detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)

emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def detect_and_predict_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detection.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    frame_clone = frame.copy()

    if len(faces) > 0:
        for (fX, fY, fW, fH) in faces:
            roi = gray_frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))  # Resize to model input size
            roi = roi.astype("float32") / 255.0  # Normalize pixel values
            roi = img_to_array(roi)  # Convert to array
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 2)
            cv2.putText(frame_clone, f"{label}: {emotion_probability:.2f}", (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    return frame_clone, faces, preds if len(faces) > 0 else None

def main():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Emotion Detection')

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to capture video. Exiting...")
            break

        frame = imutils.resize(frame, width=400)

        processed_frame, faces, preds = detect_and_predict_emotions(frame)

        cv2.imshow('Emotion Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
