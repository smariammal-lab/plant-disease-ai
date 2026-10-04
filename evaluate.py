import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------
# Load model
# -----------------------------
print("\nLoading model...")

model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)

# -----------------------------
# Load class names
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

# -----------------------------
# Check test folders
# -----------------------------
print("\nTest dataset folders:")

for class_name in class_names:
    folder = os.path.join("test_images", class_name)

    if os.path.exists(folder):
        count = len([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ])
        print(f"{class_name}: {count}")
    else:
        print(f"WARNING: Missing folder -> {class_name}")

# -----------------------------
# Load test dataset
# -----------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    "test_images",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTotal test images:", len(test_ds.file_paths))

# -----------------------------
# Prediction
# -----------------------------
print("\nRunning predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Convert to numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# Classification Report
# -----------------------------
print("\n========================================")
print("📊 TEST CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\n========================================")
print("🔍 CONFUSION MATRIX")
print("========================================")

print(cm)

# -----------------------------
# Overall Accuracy
# -----------------------------
accuracy = np.mean(
    y_true == y_pred
)

print("\n========================================")
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print("========================================")

# -----------------------------
# Correct / Wrong Predictions
# -----------------------------
correct = np.sum(y_true == y_pred)
wrong = np.sum(y_true != y_pred)

print("\nCorrect predictions:", correct)
print("Wrong predictions:", wrong)
print("Total predictions:", len(y_true))

# -----------------------------
# Class-wise Accuracy
# -----------------------------
print("\n========================================")
print("📌 CLASS-WISE ACCURACY")
print("========================================")

for i, class_name in enumerate(class_names):

    class_indices = np.where(y_true == i)[0]

    if len(class_indices) > 0:

        class_accuracy = np.mean(
            y_pred[class_indices] == i
        )

        print(
            f"{class_name}: "
            f"{class_accuracy * 100:.2f}% "
            f"({np.sum(y_pred[class_indices] == i)}/{len(class_indices)})"
        )

print("\nEvaluation completed! 🌱")