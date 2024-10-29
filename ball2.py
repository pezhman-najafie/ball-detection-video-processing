import cv2
import numpy as np

# Define video input and output paths
video_path = 'video2.mp4'
output_path = 'C:\\Users\\pezhm\\Desktop\\ball\\output_video.avi'

# Capture the input video
video_capture = cv2.VideoCapture(video_path)

# Function to detect the ball in each frame
def detect_ball(frame):
    # Convert frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30
    )
    # Draw detected circles on the frame
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

# Set up the output video writer with original video specs
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, video_capture.get(5), (int(video_capture.get(3)), int(video_capture.get(4))))

# Process each frame for ball detection
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    detect_ball(frame)
    # Display the frame with detected ball
    cv2.imshow('Video', frame)
    # Write the frame to the output video
    out.write(frame)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources
video_capture.release()
out.release()
cv2.destroyAllWindows()
