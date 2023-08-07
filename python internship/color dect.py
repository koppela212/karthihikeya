import cv2
import numpy as np

def get_color_name(rgb_color):
    # Define a dictionary of common color names and their corresponding RGB values
    color_names = {
        (255, 0, 0): "Red",
        (0, 255, 0): "Green",
        (0, 0, 255): "Blue",
        # Add more color mappings as needed
        # Example: (R, G, B): "Color Name",
        (255, 255, 255): "White",
        (0, 0, 0): "Black",
        (255, 255, 0): "Yellow",
        # Add more colors here...
    }

    closest_color = min(color_names, key=lambda x: np.linalg.norm(np.array(rgb_color) - np.array(x)))
    return color_names[closest_color], closest_color

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = frame[y, x]
        color_name, color_rgb = get_color_name(color)

        # Draw a filled rectangle with the detected color
        cv2.rectangle(frame, (x, y - 20), (x + 100, y), color_rgb, -1)
        cv2.putText(frame, color_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def main():
    global frame
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()

        if not ret:
            break

        # Convert the frame to HSV color space for easier color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for each color you want to detect
        # You can use an online color picker tool to get these values
        # For demonstration, let's use random colors within a wide range
        lower_bound = np.array([0, 70, 50])
        upper_bound = np.array([179, 255, 255])

        # Create a mask to detect all colors in the range
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Find contours of the detected colors
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Draw rectangles around the detected regions
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the original frame
        cv2.imshow("Color Detection", frame)

        cv2.setMouseCallback("Color Detection", mouse_callback)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
