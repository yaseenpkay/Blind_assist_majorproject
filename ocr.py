#text_recognition


import cv2
import pytesseract
import pyttsx3

# Path to Tesseract executable (change this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(frame):
    # Convert the frame to grayscale (optional)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    text = pytesseract.image_to_string(gray_frame)

    return text

def text_to_speech(text):
    # Initialize text-to-speech engine
    engine = pyttsx3.init()

    # Set speed (optional)
    engine.setProperty('rate', 150)  # Adjust speed (150 words per minute is the default)

    # Convert text to speech and play the audio
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Check for keypress (wait for 'p' key)
        key = cv2.waitKey(1)
        if key == ord('p'):
            # Perform OCR on the captured frame
            detected_text = perform_ocr(frame)

            # Print the detected text
            print("Detected Text:")
            print(detected_text)

            # Read the detected text aloud
            text_to_speech(detected_text)

        # Break the loop if 'q' key is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
