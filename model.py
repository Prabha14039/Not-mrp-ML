import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from product_mapping import product_mapping
from train import model_file_path
from datetime import datetime
import os


# Load the saved model
model = tf.keras.models.load_model(model_file_path)

# Function to load and preprocess image from file
def load_and_preprocess_image_from_file(file_path, target_size=(64, 64)):
    image = load_img(file_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array /= 255.0  # Normalize pixel values
    return image_array

# Function to display image
def display_image(image_array):
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function to predict and display results for a given image file
def predict_and_display(image_file):
    # Load and preprocess the image
    image = load_img(image_file, target_size=(64, 64))
    image_array = img_to_array(image)
    image_array /= 255.0  # Normalize pixel values
    
    # Predict the purchase link
    prediction = model.predict(tf.expand_dims(image_array, axis=0))
    predicted_product_index = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_product_id = list(product_mapping.keys())[predicted_product_index]
    purchase_link = product_mapping[predicted_product_id]["purchase_link"]
    
    # Display the predicted image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    return image, purchase_link


