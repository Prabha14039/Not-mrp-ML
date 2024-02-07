import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from io import BytesIO
from datetime import datetime
from product_mapping import product_mapping
import os

# Define the directory to save the trained model
save_directory = "/not@mrp/project/"

# Generate a unique file name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(product_mapping), activation='softmax')  # Output neurons equal to number of products
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Function to load and preprocess image from URL
def load_and_preprocess_image_from_url(url, target_size=(64, 64)):
    response = requests.get(url)
    image = load_img(BytesIO(response.content), target_size=target_size)
    image_array = img_to_array(image)
    image_array /= 255.0  # Normalize pixel values
    return image_array

# Prepare data for training
images = []
labels = []

for product_id, details in product_mapping.items():
    image_url = details["image_url"]
    label = list(product_mapping.keys()).index(product_id)  # One-hot encoding of labels
    images.append(load_and_preprocess_image_from_url(image_url))
    labels.append(label)

images = tf.convert_to_tensor(images)
labels = tf.keras.utils.to_categorical(labels)  # Convert labels to one-hot encoding

# Train the model
model.fit(images, labels, epochs=10)

model_file_name = f"trained_model_{timestamp}.h5"

model_file_path = os.path.join(save_directory, model_file_name)

# Save the trained model
model.save(model_file_path)
