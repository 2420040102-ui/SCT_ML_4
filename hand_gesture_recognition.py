import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Dataset path
dataset_path = "dataset"


images = []
labels = []


# Gesture classes
categories = os.listdir(dataset_path)

print("Gestures:", categories)


# Loading images

for category in categories:

    folder_path = os.path.join(dataset_path, category)

    label = categories.index(category)

    count = 0


    for gesture_folder in os.listdir(folder_path):

        gesture_path = os.path.join(folder_path, gesture_folder)


        if os.path.isdir(gesture_path):


            for img_name in os.listdir(gesture_path):


                if count == 100:
                    break


                if img_name.lower().endswith((".jpg",".jpeg",".png")):


                    img_path = os.path.join(
                        gesture_path,
                        img_name
                    )


                    image = cv2.imread(img_path)


                    if image is not None:


                        image = cv2.resize(
                            image,
                            (64,64)
                        )


                        images.append(image)

                        labels.append(label)

                        count += 1



print("Total Images:", len(images))



# Convert to numpy

X = np.array(images)

y = np.array(labels)



# Show dataset sample output

plt.figure(figsize=(10,8))


for i in range(9):


    index = random.randint(
        0,
        len(images)-1
    )


    img = images[index]


    plt.subplot(3,3,i+1)


    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    plt.imshow(img)


    plt.title(
        "Gesture: "
        + categories[y[index]]
    )


    plt.axis("off")



plt.tight_layout()


plt.savefig(
    "gesture_samples.png",
    dpi=300
)


plt.show()



# Flatten images for SVM

X = X.reshape(
    X.shape[0],
    -1
)



# Train test split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# SVM Model

model = SVC(
    kernel="linear",
    probability=True
)


print("Training Started...")


model.fit(
    X_train,
    y_train
)


print("Training Completed")



# Prediction

y_pred = model.predict(
    X_test
)



# Accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "Accuracy:",
    accuracy
)



print(
classification_report(
    y_test,
    y_pred
)
)



# Save model

joblib.dump(
    model,
    "svm_gesture_model.pkl"
)


print("Model Saved")



# Test image prediction


test_img_path = os.path.join(
    dataset_path,
    categories[0],
    os.listdir(
        os.path.join(
            dataset_path,
            categories[0]
        )
    )[0]
)



img = cv2.imread(
    test_img_path
)



image = cv2.resize(
    image,
    (32,32)
)



prediction = model.predict(
    resize_img.reshape(1,-1)
)



predicted_gesture = categories[
    prediction[0]
]


print(
    "Predicted Gesture:",
    predicted_gesture
)



# Prediction output image


img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


plt.imshow(
    img_rgb
)


plt.title(
    "Predicted Gesture: "
    + predicted_gesture
)


plt.axis("off")


plt.savefig(
    "prediction_output.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()