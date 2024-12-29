import cv2
import numpy as np
import mediapipe as mp
import mouse
from tensorflow.keras.models import load_model

# Load the trained gesture recognition model
model = load_model("gesture_model.h5")

# Gesture labels
gesture_labels = {
    0: "leftclick",
    1: "rightclick",
    2: "cursor",
    3: "scrollup",
    4: "scrolldown",
    5: "drag",
}

# Screen resolution and scaling factor
SCREEN_WIDTH, SCREEN_HEIGHT = 1366, 768  # Change to your screen resolution
CAM_WIDTH, CAM_HEIGHT = 640, 480
SCALING_FACTOR = 2.0  # Multiply hand movement by this factor for extended cursor range

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Smoothing factor and previous coordinates
alpha = 0.3  # Smoothing factor for cursor movement
cursor_x, cursor_y = 0, 0

# Gesture state variables
dragging = False

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

print("Starting gesture-based mouse control. Press 'q' to exit.")

# Add a flag to track right-click state
right_click_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Set a default gesture name in case no hands are detected
    gesture_name = "No Gesture"

    # Detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks as input for the model
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Normalize coordinates between 0 and 1
            
            # Predict gesture
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)
            gesture = np.argmax(prediction)
            gesture_name = gesture_labels[gesture]

            # Map hand position to scaled screen coordinates
            ind_x, ind_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
            mapped_x = int(np.interp(ind_x, [0, 1], [0, SCREEN_WIDTH * SCALING_FACTOR]))
            mapped_y = int(np.interp(ind_y, [0, 1], [0, SCREEN_HEIGHT * SCALING_FACTOR]))

            # Apply exponential smoothing for smooth cursor movement
            cursor_x = int(alpha * mapped_x + (1 - alpha) * cursor_x)
            cursor_y = int(alpha * mapped_y + (1 - alpha) * cursor_y)

            # Perform actions based on gesture
            if gesture_name == "cursor":
                mouse.move(cursor_x // SCALING_FACTOR, cursor_y // SCALING_FACTOR)
            elif gesture_name == "leftclick":
                mouse.click()
                cv2.putText(frame, "Left Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture_name == "rightclick":
                if not right_click_done:
                    mouse.click(button='right')  # Perform a single right-click
                    right_click_done = True  # Set the flag to prevent further clicks
                    cv2.putText(frame, "Right Click", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture_name != "rightclick":
                # Reset the right-click flag if the gesture is no longer "rightclick"
                right_click_done = False

            if gesture_name == "scrollup":
                mouse.wheel(1)
                cv2.putText(frame, "Scroll Up", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            elif gesture_name == "scrolldown":
                mouse.wheel(-1)
                cv2.putText(frame, "Scroll Down", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            elif gesture_name == "drag":
                mouse.move(cursor_x // SCALING_FACTOR, cursor_y // SCALING_FACTOR)  # Update cursor position while dragging
                if not dragging:
                    mouse.press()
                    dragging = True
                cv2.putText(frame, "Dragging", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # Release drag if gesture is no longer "drag"
                if dragging:
                    mouse.release()
                    dragging = False

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Mouse Control", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
