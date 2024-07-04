import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# Function to load data
def load_data(landslide_dir, non_landslide_dir, image_size=(128, 128)):
    landslide_images = []
    non_landslide_images = []

    for filename in os.listdir(landslide_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(landslide_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            landslide_images.append(image)

    for filename in os.listdir(non_landslide_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(non_landslide_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            non_landslide_images.append(image)

    landslide_images = np.array(landslide_images) / 255.0
    non_landslide_images = np.array(non_landslide_images) / 255.0

    return landslide_images, non_landslide_images

landslide_dir = '/content/drive/MyDrive/landslide'
non_landslide_dir = '/content/drive/MyDrive/non landslide'

landslide_images, non_landslide_images = load_data(landslide_dir, non_landslide_dir)

landslide_labels = np.ones(len(landslide_images))  # 1 for landslide
non_landslide_labels = np.zeros(len(non_landslide_images))  # 0 for non-landslide

all_images = np.concatenate([landslide_images, non_landslide_images])
all_labels = np.concatenate([landslide_labels, non_landslide_labels])

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=4)

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train multiple CNN models
num_models = 3
trained_models = []

for i in range(num_models):
    model = create_model(input_shape=X_train[0].shape)
    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))
    trained_models.append(model)

# Make predictions with each model
predictions = np.zeros((len(X_test), num_models))
for i, model in enumerate(trained_models):
    predictions[:, i] = np.squeeze(model.predict(X_test))

# Aggregate predictions using voting ensemble
ensemble_predictions = (np.sum(predictions, axis=1) > num_models / 2).astype(int)

accuracy = accuracy_score(y_test, ensemble_predictions)
classification_report = classification_report(y_test, ensemble_predictions)

mcc = matthews_corrcoef(y_test, ensemble_predictions)

# Calculate Intersection over Union (IoU)
conf_matrix = confusion_matrix(y_test, ensemble_predictions)
true_positives = conf_matrix[1, 1]
false_positives = conf_matrix[0, 1]
false_negatives = conf_matrix[1, 0]

iou = true_positives / (true_positives + false_positives + false_negatives)
print(f"Matthews Correlation Coefficient (MCC): {mcc}")
print(f"Intersection over Union (IoU): {iou}")

print(f"Ensemble Model Accuracy: {accuracy * 100}%")
print(f"Ensemble Model Classification Report:\n{classification_report}")
