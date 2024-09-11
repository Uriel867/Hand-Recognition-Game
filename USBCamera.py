import cv2

for i in range(5):  # Check multiple indexes to see if your camera is listed
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
    else:
        print(f"Camera {i} is not available")