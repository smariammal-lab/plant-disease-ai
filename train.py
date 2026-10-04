import tensorflow as tf
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_DIR = "dataset"
MODEL_DIR = "model"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

print("\nNumber of classes:", len(class_names))

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

# ==========================================
# 3. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1)
])

# ==========================================
# 4. PREPROCESSING
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# 5. CALCULATE CLASS WEIGHTS
# ==========================================

# Your current dataset counts:
# Bacterial Spot = 2127
# Early Blight   = 1000
# Healthy        = 1591
# Late Blight    = 1909
# Leaf Mold      = 952

class_counts = []

for class_name in class_names:
    class_path = os.path.join(DATASET_DIR, class_name)

    count = len([
        file for file in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, file))
    ])

    class_counts.append(count)

print("\nClass image counts:")
for name, count in zip(class_names, class_counts):
    print(f"{name}: {count}")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.concatenate([
        np.full(count, i)
        for i, count in enumerate(class_counts)
    ])
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
for i, weight in class_weights.items():
    print(f"{class_names[i]}: {weight:.3f}")

# ==========================================
# 6. MOBILE NET V2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ==========================================
# 7. BUILD MODEL
# ==========================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 8. COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 9. CALLBACKS
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "plant_disease_model.keras"
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\n========================================")
print("Starting Training...")
print("========================================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# ==========================================
# 11. FINAL EVALUATION
# ==========================================

print("\n========================================")
print("Final Validation Result")
print("========================================")

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\nValidation Loss:", loss)
print("Validation Accuracy:", accuracy * 100, "%")

# ==========================================
# 12. SAVE MODEL
# ==========================================

model.save(model_path)

print("\nModel saved successfully!")
print("Location:", model_path)

# ==========================================
# 13. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "training_accuracy.png"
    )
)

plt.close()

# ==========================================
# 14. LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "training_loss.png"
    )
)

plt.close()

# ==========================================
# 15. COMPLETE
# ==========================================

print("\n========================================")
print("TRAINING COMPLETED!")
print("========================================")

print("\nFiles created:")

print("1. model/plant_disease_model.keras")
print("2. model/training_accuracy.png")
print("3. model/training_loss.png")
print("4. class_names.json")

print("\nYour Plant Disease AI model is ready! 🌱")