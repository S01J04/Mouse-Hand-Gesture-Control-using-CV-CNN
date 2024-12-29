import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Output directory for dataset
output_dir = "gesture_dataset"
os.makedirs(output_dir, exist_ok=True)

# List of gesture labels
gesture_labels = {
    0: "leftclick",
    1: "rightclick",
    2: "cursor",
    3: "scrollup",
    4: "scrolldown",
}

print("Gesture Labels:")
for label, gesture in gesture_labels.items():
    print(f"{label}: {gesture}")

# Choose the gesture label to collect
gesture_label = int(input("Enter the gesture label you want to collect: "))
if gesture_label not in gesture_labels:
    print("Invalid label. Exiting...")
    exit()

# Create a folder for the selected gesture
gesture_folder = os.path.join(output_dir, f"{gesture_label}_{gesture_labels[gesture_label].replace(' ', '_')}")
os.makedirs(gesture_folder, exist_ok=True)

# Initialize data storage
data = []
labels = []

# Start video capture
cap = cv2.VideoCapture(0)

print(f"Collecting data for gesture: {gesture_labels[gesture_label]}")
print("Press 'q' to stop collecting data.")

image_count = 0  # Counter for saved images

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    # Extract landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Store landmarks as a flattened array
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Normalize coordinates between 0 and 1
            data.append(landmarks)
            labels.append(gesture_label)

            # Save the image
            image_path = os.path.join(gesture_folder, f"image_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            image_count += 1

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.putText(frame, f"Gesture: {gesture_labels[gesture_label]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Samples: {len(data)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dataset Collection", frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the dataset
data = np.array(data)
labels = np.array(labels)

data_file = os.path.join(gesture_folder, f"hand_landmarks.npy")
labels_file = os.path.join(gesture_folder, f"gesture_labels.npy")

np.save(data_file, data)
np.save(labels_file, labels)

print(f"Dataset saved for gesture '{gesture_labels[gesture_label]}'.")
print(f"Landmarks: {data_file}")
print(f"Labels: {labels_file}")
print(f"Images saved in folder: {gesture_folder}")

# Release resources
cap.release()
cv2.destroyAllWindows()
