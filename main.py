import cv2
import serial
import pygame

# Load pedestrian detection cascade classifier
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')


pygame.init()

# Load the audio file (WAV format)
audio_file = "Kulpi 75.wav"
pygame.mixer.music.load(audio_file)

# Function to play the audio
def play_audio():
    pygame.mixer.music.play()

# Optimized function to detect pedestrians in an image
def detect_pedestrians(image):
    # Preprocessing: convert to grayscale, apply GaussianBlur, and improve contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # Detect pedestrians with optimized parameters
    pedestrians = pedestrian_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Adjust scaling for multi-scale detection
        minNeighbors=4,   # Reduce for better distant detection
        minSize=(40, 40)  # Adjust minimum pedestrian size
    )

    # Draw optimized rectangles for detected pedestrians
    for (x, y, w, h) in pedestrians:
        thickness = max(2, int(0.01 * (w + h)))  # Thickness proportional to box size
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness)

    # Display message indicating pedestrians detected
    if len(pedestrians) > 0:
        message = f"Pedestrians Detected: {len(pedestrians)}"
        cv2.putText(image, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image, len(pedestrians)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        break

    # Detect pedestrians in the frame
    frame_with_pedestrians, num_pedestrians = detect_pedestrians(frame)

    # Display the frame with pedestrian detection
    cv2.imshow('Pedestrian Detection', frame_with_pedestrians)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
