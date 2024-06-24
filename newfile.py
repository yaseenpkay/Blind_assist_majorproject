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

# Callback function for non-blocking speech synthesis
def on_end(name, completed):
    pass  # Add any additional logic if needed

# Function to speak the detected object
def speak_detected_object(class_name, obj_id, distance):
    speech_text = f"New {class_name}  with distance {distance} metres"
    print(speech_text)
    engine.say(speech_text)
    engine.runAndWait()

# Function to run text-to-speech in a separate thread
def run_tts(class_name, obj_id, distance):
    threading.Thread(target=speak_detected_object, args=(class_name, obj_id, distance)).start()

# Start webcam
cap = cv2.VideoCapture(0)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
center_x = int(frame_width // 2)
center_y = int(frame_height // 2)
cap.set(3, 640)
cap.set(4, 480)

# Load model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

object_dimensions = {
    "person": 0.45,"bicycle": 1.76,"car": 4.5,"motorbike": 2.0,"aeroplane": 15.0,"bus": 12.0,"train": 20.0,"truck": 8.0,"boat": 3.0,"traffic light": 0.5,"fire hydrant": 0.3,
    "stop sign": 0.5,"parking meter": 0.3,
    "bench": 1.5,"bird": 0.1,"cat": 0.45,"dog": 0.55,"horse": 1.5,"sheep": 1.0,"cow": 1.5,"elephant": 4.0,"bear": 1.5,"zebra": 1.0,"giraffe": 4.0,
    "backpack": 0.5,"umbrella": 0.7,"handbag": 0.4, "tie": 0.2,"suitcase": 0.6, "frisbee": 0.3,"skis": 1.5,
    "snowboard": 1.5,"sports ball": 0.3,"kite": 0.5,
    "baseball bat": 0.8, "baseball glove": 0.3, "skateboard": 1.0,"surfboard": 2.0, "tennis racket": 0.5,"bottle": 0.3,
    "wine glass": 0.2,"cup": 0.15,"fork": 0.15,"knife": 0.2,"spoon": 0.15,"bowl": 0.25,"banana": 0.2,
    "apple": 0.07,"sandwich": 0.15,"orange": 0.08,"broccoli": 0.15,"carrot": 0.1,"hot dog": 0.2,"pizza": 0.25,"donut": 0.1,"cake": 0.3,"chair": 0.5,"sofa": 1.8,
    "pottedplant": 0.4,"bed": 2.0,"diningtable": 1.0,"toilet": 0.6,"tvmonitor": 1.0,"laptop": 0.4,"mouse": 0.1,"remote": 0.1,"keyboard": 0.3,"cell phone": 0.2,"microwave": 0.6,"oven": 0.6,"toaster": 0.3,"sink": 0.6,"refrigerator": 0.8,"book": 0.15,"clock": 0.2,"vase": 0.25,"scissors": 0.2,"teddy bear": 0.3,"hair drier": 0.2,"toothbrush": 0.1
}


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

            real_width = float(object_dimensions[class_name])

            camera_width = x2 - x1
            distance_long = (real_width * frame_width) / camera_width
            distance = "{:.2f}".format(distance_long)

            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2

            camera_middle_x = frame_width // 2
            camera_middle_y = frame_height // 2

            vector_x = obj_center_x - camera_middle_x
            vector_y = obj_center_y - camera_middle_y

            # Check if the box has an ID
            if box.id is not None:
                obj_id = int(box.id[0])  # Unique ID for the detected object

                # Check if the object ID is new for the current class
                if obj_id not in previous_ids[class_name]:
                    # Update the record of IDs for this class
                    previous_ids[class_name].add(obj_id)

                    # Run text-to-speech in a separate thread
                    run_tts(class_name, obj_id, distance)

            # Add bounding box coordinates to the current_boxes dictionary
            current_boxes[class_name].add((x1, y1, x2, y2))

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{class_name}: {distance}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


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
