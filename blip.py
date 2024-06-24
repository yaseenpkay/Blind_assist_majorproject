# from transformers import pipeline
# from PIL import Image
# import cv2
# import pyttsx3


# # Instantiate the pipeline for image captioning
# image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# def capture_image():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         cv2.imshow('Webcam', frame)
#         if cv2.waitKey(1) == ord('p'):
#             cv2.imwrite('captured_image.jpg', frame)
#             break
#     cap.release()
#     cv2.destroyAllWindows()


# def text_to_speech(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()

# def main():
#     capture_image()
#     image = Image.open('captured_image.jpg')
#     captions = image_to_text(image)
#     for caption in captions:
#         # Extract the caption text from the dictionary
#         caption_text = caption.get('generated_text', '')
#         text_to_speech(caption_text)
#         print("caption is : ", caption_text)


# if __name__ == "__main__":
#     main()


# from transformers import pipeline
# from PIL import Image
# import cv2
# import pyttsx3

# # Instantiate the pipeline for image captioning
# image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# def capture_image():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         cv2.imshow('Webcam', frame)
#         if cv2.waitKey(1) == ord('p'):
#             cv2.imwrite('captured_image.jpg', frame)
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# def text_to_speech(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()

# def main():
#     while True:
#         capture_image()
#         image = Image.open('captured_image.jpg')
#         captions = image_to_text(image)
#         for caption in captions:
#             caption_text = caption.get('generated_text', '')
#             text_to_speech(caption_text)
#             print("caption is : ", caption_text)
        
#         # Ask if the user wants to capture another image
#         choice = input("Do you want to capture another image? (y/n): ")
#         if choice.lower() != 'y':
#             break

# if __name__ == "__main__":
#     main()


from transformers import pipeline
from PIL import Image
import cv2
import pyttsx3

# Instantiate the pipeline for image captioning
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)
        if key == ord('p'):
            cv2.imwrite('captured_image.jpg', frame)
            cap.release()
            cv2.destroyAllWindows()
            return True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    while True:
        if capture_image():
            image = Image.open('captured_image.jpg')
            captions = image_to_text(image)
            for caption in captions:
                caption_text = caption.get('generated_text', '')
                text_to_speech(caption_text)
                print("caption is : ", caption_text)

if __name__ == "__main__":
    main()

