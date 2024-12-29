import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Gesture labels
gesture_labels = {
    0: "leftclick",
    1: "rightclick",
    2: "cursor",
    3: "scrollup",
    4: "scrolldown",
}

# Load dataset
def load_dataset(base_dir="gesture_dataset"):
    
    data = []
    labels = []
    for label, gesture in gesture_labels.items():
        folder = os.path.join(base_dir, f"{label}_{gesture.replace(' ', '_')}")
        landmarks_path = os.path.join(folder, "hand_landmarks.npy")
        labels_path = os.path.join(folder, "gesture_labels.npy")
        
        if os.path.exists(landmarks_path) and os.path.exists(labels_path):
            data.extend(np.load(landmarks_path))
            labels.extend(np.load(labels_path))
        else:
            print(f"Data for gesture {gesture} not found.")
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Load the dataset
print("Loading dataset...")
data, labels = load_dataset()
print(f"Dataset loaded. Total samples: {len(data)}")

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(gesture_labels))
y_test = to_categorical(y_test, num_classes=len(gesture_labels))

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(data.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(gesture_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Save the trained model
model.save("gesture_model.h5")
print("Model saved as gesture_model.h5")
