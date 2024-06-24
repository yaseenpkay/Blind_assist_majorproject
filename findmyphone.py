import cv2
import numpy as np

# Define the color ranges for different colors in HSV format
color_ranges = {
    'red':      ([0, 100, 100], [10, 255, 255]),       # Red
    'green':    ([40, 100, 100], [80, 255, 255]),      # Green
    'blue':     ([100, 100, 100], [140, 255, 255]),    # Blue
    'yellow':   ([20, 100, 100], [40, 255, 255]),      # Yellow
    'orange':   ([10, 100, 100], [20, 255, 255]),      # Orange
    'purple':   ([140, 100, 100], [160, 255, 255]),    # Purple
    'pink':     ([160, 100, 100], [180, 255, 255]),    # Pink
    'cyan':     ([80, 100, 100], [100, 255, 255]),     # Cyan
    'magenta':  ([140, 100, 100], [160, 255, 255]),    # Magenta
    'white':    ([0, 0, 200], [180, 50, 255]),         # White
    'black':    ([0, 0, 0], [180, 255, 30]),           # Black
    'gray':     ([0, 0, 100], [180, 30, 200]),         # Gray
    'brown':    ([10, 100, 100], [20, 255, 150]),      # Brown
    'beige':    ([10, 50, 150], [30, 200, 255]),       # Beige
    'olive':    ([40, 50, 100], [80, 200, 200]),       # Olive
    'maroon':   ([0, 100, 50], [10, 255, 150]),        # Maroon
}

# Function to find the color of an object in the image
def find_color(image, color):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the color range
    lower_bound = np.array(color_ranges[color][0])
    upper_bound = np.array(color_ranges[color][1])
    # Threshold the HSV image to get only the desired color
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    # Find contours of the color regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If contours are found, draw a bounding box around the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Find colors in the frame
    for color in color_ranges:
        frame = find_color(frame, color)

    # Display the resulting frame
    cv2.imshow('Color Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
