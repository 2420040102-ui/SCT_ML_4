import warnings
warnings.filterwarnings("ignore")

import os, cv2, random, joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

dataset_path = "dataset"

images, labels = [], []

categories = sorted(os.listdir(dataset_path))

print("Gestures:", categories)

for label, category in enumerate(categories):

    count = 0

    for root, _, files in os.walk(os.path.join(dataset_path, category)):

        for file in files:

            if count >= 30:
                break

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                img = cv2.imread(os.path.join(root, file))

                if img is not None:

                    img = cv2.resize(img, (32,32))

                    images.append(img)
                    labels.append(label)

                    count += 1

print("Total Images:", len(images))

X = np.array(images)
y = np.array(labels)

# Sample Images

plt.figure(figsize=(8,8))

for i in range(9):

    idx = random.randint(0, len(images)-1)

    plt.subplot(3,3,i+1)

    plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

    plt.title(categories[y[idx]])

    plt.axis("off")

plt.tight_layout()
plt.savefig("gesture_samples.png")
plt.close()

# Training

X = X.reshape(len(X), -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Started...")

model = LinearSVC(max_iter=3000)

model.fit(X_train, y_train)

print("Training Completed")

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

joblib.dump(model, "svm_gesture_model.pkl")

print("Model Saved")

# Prediction Output

img = images[0]

gesture = categories[
    model.predict(img.reshape(1,-1))[0]
]

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.title(f"Predicted Gesture: {gesture}")

plt.axis("off")

plt.savefig(
    "prediction_output.png",
    bbox_inches="tight"
)

plt.close()

print("Prediction image saved")