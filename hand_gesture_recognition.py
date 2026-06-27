import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


dataset_path = "dataset"


images = []
labels = []


categories = os.listdir(dataset_path)


print("Gestures:", categories)





# -----------------------------
# Loading Images (Fast Version)
# -----------------------------

categories = os.listdir(dataset_path)

print("Gestures:", categories)


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


                img_path = os.path.join(gesture_path, img_name)


                if img_name.lower().endswith((".jpg",".jpeg",".png")):


                    image = cv2.imread(img_path)


                    if image is not None:

                        image = cv2.resize(image,(64,64))


                        images.append(image)

                        labels.append(label)


                        count += 1



print("Total Images:", len(images))


X = np.array(images)
y = np.array(labels)



X = X.reshape(X.shape[0],-1)



X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = SVC(kernel="linear")


print("Training Started...")


model.fit(
    X_train,
    y_train
)


print("Training Completed")



y_pred = model.predict(X_test)



print(
    "Accuracy:",
    accuracy_score(y_test,y_pred)
)



print(
classification_report(
    y_test,
    y_pred
)
)



# Test image

test_img_path = os.path.join(
    dataset_path,
    categories[0],
    os.listdir(
        os.path.join(dataset_path,categories[0])
    )[0]
)



img = cv2.imread(test_img_path)


img_resize = cv2.resize(img,(64,64))


prediction = model.predict(
    img_resize.reshape(1,-1)
)



print(
    "Predicted Gesture:",
    categories[prediction[0]]
)



img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


plt.imshow(img)
plt.axis("off")
plt.show()