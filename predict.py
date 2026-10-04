import tensorflow as tf
import json
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model/plant_disease_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Get image path
image_path = input("Enter image path: ").strip()

# Load and resize image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Prediction
predictions = model.predict(image_array, verbose=0)

# Get predicted class
predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

# Get confidence
confidence = predictions[0][predicted_index] * 100

print("\n==============================")
print("🌱 Plant Disease Prediction")
print("==============================")
print("Prediction :", predicted_class)
print("Confidence :", f"{confidence:.2f}%")
print("==============================")