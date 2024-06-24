#object_detetcion

from ultralytics import YOLO
import cv2
import math
import pyttsx3
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set properties, if needed
engine.setProperty("rate", 150)  # Speed of speech, words per minute
engine.setProperty("volume", 1.0)  # Volume level (0.0 to 1.0)

video_url = "http://192.168.1.2:8080/video"


# Callback function for non-blocking speech synthesis
def on_end(name, completed):
    pass  # Add any additional logic if needed

# Function to speak the detected object
def speak_detected_object(class_name, obj_id,):
    speech_text = f"{class_name} Rupees"
    print(speech_text)
    engine.say(speech_text)
    engine.runAndWait()

# Function to run text-to-speech in a separate thread
def run_tts(class_name, obj_id):
    threading.Thread(target=speak_detected_object, args=(class_name, obj_id, )).start()

# Start webcam
cap = cv2.VideoCapture(0)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
center_x = int(frame_width // 2)
center_y = int(frame_height // 2)
cap.set(3, 640)
cap.set(4, 480)

# Load model
model = YOLO("currency.pt")

# Object classes
# Object classes
classNames = ["10", "20", "50", "100", "200", "500", "2000",
                  ]



previous_ids = {class_name: set() for class_name in classNames}

while True:
    success, img = cap.read()
    results = model.track(img, show=True, tracker="bytetrack.yaml", conf=0.3, iou=0.5, persist=True)

    # Initialize a dictionary to store bounding box coordinates for each class
    current_boxes = {class_name: set() for class_name in classNames}

    # Loop through the results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract information from the box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]


            # Check if the box has an ID
            if box.id is not None:
                obj_id = int(box.id[0])  # Unique ID for the detected object

                # Check if the object ID is new for the current class
                if obj_id not in previous_ids[class_name]:
                    # Update the record of IDs for this class
                    previous_ids[class_name].add(obj_id)

                    # Run text-to-speech in a separate thread
                    run_tts(class_name, obj_id, )

            # Add bounding box coordinates to the current_boxes dictionary
            current_boxes[class_name].add((x1, y1, x2, y2))

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{class_name}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    # Remove IDs of objects that have left the frame
    for class_name in classNames:
        for obj_id in previous_ids[class_name].copy():
            # Check if the object's bounding box is not present in the current frame
            if not any(bbox for bbox in current_boxes[class_name] if bbox[2] > 0 and bbox[3] > 0):
                previous_ids[class_name].remove(obj_id)  # Remove the ID from the list

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
