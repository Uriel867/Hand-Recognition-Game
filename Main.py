import cv2
import mediapipe as mp
import numpy as np


def HandGunGame():
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize MediaPipe Drawing module for drawing landmarks
    mp_drawing = mp.solutions.drawing_utils

    # Open a video capture object (1 for the USB camera or 0 for the default)
    cap = cv2.VideoCapture(1)

    # Tolerance for detecting overlapping landmarks
    tolerance = 0.042

    # Circle properties
    circle_x = 100  # Start position for the circle
    circle_y = 100  # Start position for the circle
    circle_radius = 10
    speed_x = 2
    speed_y = 2
    direction = [1, 1]

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the position of the 6th landmark to reset circle position
                hand_circle_x = int(hand_landmarks.landmark[6].x * frame.shape[1])
                hand_circle_y = int(hand_landmarks.landmark[6].y * frame.shape[0])

                # Check if the landmarks are on top of each other
                if (abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[6].x) < tolerance and
                        abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[6].y) < tolerance):

                    print("The landmarks are on top of each other.")
                    direction = get_finger_direction(hand_landmarks, frame)
                    # Optional: Reset circle position to the hand landmark position
                    circle_x, circle_y = hand_circle_x, hand_circle_y

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Update circle position for animation
        circle_x += speed_x * direction[0]
        circle_y += speed_y * direction[1]



        # Draw the animated circle at the updated position
        cv2.circle(frame, (int(circle_x), int(circle_y)), circle_radius, (0, 255, 0), -1)

        # Display the frame with hand landmarks and the animated circle
        cv2.imshow('Hand Recognition', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def get_finger_direction(hand_landmarks, frame):
    # Get coordinates of the tip of the middle finger (landmark 8)
    tip_x = hand_landmarks.landmark[8].x * frame.shape[1]
    tip_y = hand_landmarks.landmark[8].y * frame.shape[0]

    # Get coordinates of the base of the middle finger (landmark 7)
    base_x = hand_landmarks.landmark[7].x * frame.shape[1]
    base_y = hand_landmarks.landmark[7].y * frame.shape[0]

    # Calculate the direction vector
    direction_vector = np.array([tip_x - base_x, tip_y - base_y])

    # Normalize the direction vector
    direction_magnitude = np.linalg.norm(direction_vector)
    if direction_magnitude != 0:
        normalized_direction = direction_vector / direction_magnitude
    else:
        normalized_direction = np.array([0, 0])

    return normalized_direction


HandGunGame()
