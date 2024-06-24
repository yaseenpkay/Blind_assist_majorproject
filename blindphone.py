import math
import cv2
import threading
import pyttsx3
from ultralytics import YOLO
import pytesseract
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import keyboard
import speech_recognition as sr


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

video_url = "http://192.168.1.2:8080/video"


# Initialize the text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# Set properties, if needed
engine.setProperty('voice', voices[1].id)
engine.setProperty("rate", 150)  # Speed of speech, words per minute
engine.setProperty("volume", 1.0)  # Volume level (0.0 to 1.0)

# Callback function for non-blocking speech synthesis
def on_end(name, completed):
    pass  # Add any additional logic if needed

# Function to speak the detected object
def speak_detected_object(class_name, obj_id, distance):
    speech_text = f"New {class_name} with distance {distance} metres"
    print(speech_text)
    engine.say(speech_text)
    engine.runAndWait()

# Function to run text-to-speech in a separate thread
def run_tts(class_name, obj_id, distance):
    threading.Thread(target=speak_detected_object, args=(class_name, obj_id, distance)).start()

def speak_detected_currency(class_name, obj_id):
    speech_text = f" {class_name} Rupees"
    print(speech_text)
    engine.say(speech_text)
    engine.runAndWait()    

def run_tts_currency(class_name, obj_id):
    threading.Thread(target=speak_detected_currency, args=(class_name, obj_id)).start()    




# Function to perform speech recognition
def perform_speech_recognition():
    recognizer = sr.Recognizer()

    print("Recording... Press 'v' again to stop.")
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio_data).lower()
            print("You said:", command)
            engine.say("You said:" + command)
            engine.runAndWait()
            if command == "object detection":
                print("Running Object Detection")
                engine.say("Running Object Detection")
                engine.runAndWait()
                perform_object_detection()
            elif command == "text recognition":
                print("Running Text recoginition")
                engine.say("Running Text recognition")
                engine.runAndWait()
                perform_text_recognition()
            elif command == "scene description":
                print("Running Scene description")
                engine.say("Running Scene Description")
                engine.runAndWait()
                perform_image_captioning()

            elif command == "currency detection":
                print("Running currency detection")
                engine.say("Running currency detection")
                engine.runAndWait()
                perform_currency_detection()

            else:
                print("Invalid command. Please try again.")
                engine.say("Invalid command. Please try again.")
                engine.runAndWait()
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")
            engine.say("Sorry, I didn't catch that. Please try again.")
            engine.runAndWait()
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Function for object detection
def perform_object_detection():
    # Start webcam
    cap = cv2.VideoCapture(video_url)
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

        # Check if 'x' key is pressed to stop object detection
        if keyboard.is_pressed('x'):
            break

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

def perform_currency_detection():
    cap = cv2.VideoCapture(video_url)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    center_x = int(frame_width // 2)
    center_y = int(frame_height // 2)
    cap.set(3, 640)
    cap.set(4, 480)

    # Load model
    model = YOLO("best.pt")

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
                        run_tts_currency(class_name, obj_id )

                # Add bounding box coordinates to the current_boxes dictionary
                current_boxes[class_name].add((x1, y1, x2, y2))

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, f"{class_name}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Check if 'x' key is pressed to stop object detection
        if keyboard.is_pressed('x'):
            break        


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


# Function for text recognition

def perform_text_recognition():
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

    # Open the webcam
    cap = cv2.VideoCapture(video_url)


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

        # Check if 'x' key is pressed to stop text recognition
        if keyboard.is_pressed('x'):
            break

        # Break the loop if 'q' key is pressed
        elif key == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function for image captioning
# def perform_image_captioning():
#     model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     max_length = 16
#     num_beams = 4
#     gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

#     def predict_caption(image):
#         pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
#         pixel_values = pixel_values.to(device)

#         output_ids = model.generate(pixel_values, **gen_kwargs)

#         preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#         preds = [pred.strip() for pred in preds]
#         print("caption is : ", preds)
#         return preds

#     def capture_image():
#         cap = cv2.VideoCapture(0)
#         while True:
#             ret, frame = cap.read()
#             cv2.imshow('Webcam', frame)
#             if cv2.waitKey(1) == ord('p'):
#                 cv2.imwrite('captured_image.jpg', frame)
#                 break
#         cap.release()
#         cv2.destroyAllWindows()

#     def text_to_speech(text):
#         engine = pyttsx3.init()
#         engine.say(text)
#         engine.runAndWait()

#     capture_image()
#     image = Image.open('captured_image.jpg')
#     captions = predict_caption([image])
#     for caption in captions:
#         text_to_speech(caption)

def perform_image_captioning():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def text_to_speech(text):
         engine = pyttsx3.init()
         engine.say(text)
         engine.runAndWait()

    # Start the webcam
    cap = cv2.VideoCapture(video_url)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)

        # Check for 'p' keypress to capture image
        if cv2.waitKey(1) == ord('p'):
            cv2.imwrite('captured_image.jpg', frame)

            # Perform image captioning
            image = Image.open('captured_image.jpg')
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            output_ids = model.generate(pixel_values, **gen_kwargs)

            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            caption = " ".join(preds)
            text_to_speech(caption)

        # Check for 'x' keypress to exit
        if keyboard.is_pressed('x'):
            break

    cap.release()
    cv2.destroyAllWindows()



# Main function
def main():

    engine.say("How may I assist you? Press and hold the button to record")
    engine.runAndWait()
    print("How may I assist you? Press and hold the button to record")

    is_recording = False

    while True:
        if keyboard.is_pressed('v'):
            if not is_recording:
                perform_speech_recognition()
                is_recording = True
        else:
            is_recording = False

        if keyboard.is_pressed('x'):
            break

if __name__ == "__main__":
    main()
