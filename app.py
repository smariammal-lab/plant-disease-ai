from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

app = Flask(__name__)

# ==============================
# MODEL SETTINGS
# ==============================

MODEL_PATH = "model/plant_disease_model.keras"
IMG_SIZE = (224, 224)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print("Classes:", class_names)


# ==============================
# DISEASE INFORMATION
# ==============================

disease_info = {

    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot",
        "advisory":
            "Remove infected leaves and avoid overhead watering. "
            "Maintain good air circulation around the plant."
    },

    "Tomato_Early_blight": {
        "name": "Early Blight",
        "advisory":
            "Remove affected leaves, avoid wetting the foliage, "
            "and maintain proper plant spacing."
    },

    "Tomato_Late_blight": {
        "name": "Late Blight",
        "advisory":
            "Remove severely affected leaves and fruits, "
            "avoid overhead watering, and improve air circulation."
    },

    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold",
        "advisory":
            "Improve ventilation, reduce excess humidity, "
            "and avoid prolonged leaf wetness."
    },

    "Tomato_healthy": {
        "name": "Healthy",
        "advisory":
            "The leaf appears healthy. Continue regular monitoring "
            "and good crop management."
    }
}


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# PREDICTION
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image exists
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    # Check filename
    if file.filename == "":
        return jsonify({
            "error": "Please select an image"
        }), 400

    try:

        # ==============================
        # LOAD IMAGE
        # ==============================

        image = Image.open(file).convert("RGB")

        # Resize image
        image = image.resize(IMG_SIZE)

        # Convert image to NumPy array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)


        # ==============================
        # MODEL PREDICTION
        # ==============================

        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get highest probability class
        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        # Confidence percentage
        confidence = float(
            predictions[0][predicted_index] * 100
        )


        # ==============================
        # GET DISEASE INFORMATION
        # ==============================

        info = disease_info.get(
            predicted_class,
            {
                "name": predicted_class,
                "advisory":
                    "Please consult an agricultural expert "
                    "for further diagnosis."
            }
        )


        # ==============================
        # SEND RESULT TO WEBSITE
        # ==============================

        return jsonify({

            "prediction": predicted_class,

            "disease_name": info["name"],

            "confidence": round(
                confidence,
                2
            ),

            "advisory": info["advisory"]

        })


    except Exception as e:

        print("Prediction Error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ==============================
# RUN FLASK SERVER
# ==============================


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )