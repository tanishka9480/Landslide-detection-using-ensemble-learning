from google.colab import files
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipywidgets as widgets

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

def predict_uploaded_image(b):
    file_upload = files.upload()

    image_path = list(file_upload.keys())[0]

    image = preprocess_image(image_path)

    prediction = model.predict(np.expand_dims(image, axis=0))
    if prediction[0][0] > 0.5:
        predicted_category = "Landslide"
    else:
        predicted_category = "Non-Landslide"

    plt.imshow(image)
    plt.title(f"Predicted Category: {predicted_category}")
    plt.axis('off')
    plt.show()

upload_button = widgets.FileUpload()
upload_button.observe(predict_uploaded_image, names='data')

display(upload_button)
