import os
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------
# Model configuration
# -------------------------
MODELS_DIR = "."
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "waste_model.tflite")

# -------------------------
# Try loading the model
# -------------------------
interpreter = None
input_details = None
output_details = None
MODEL_READY = False

try:
    if os.path.exists(TFLITE_MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        MODEL_READY = True
        print(f"✅ Loaded TFLite model: {TFLITE_MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found: {TFLITE_MODEL_PATH}. Using fallback mode.")
except Exception as e:
    print(f"⚠️ Failed to load TFLite model: {e}")
    MODEL_READY = False


# -------------------------
# Labels
# -------------------------
def load_labels(path: str = "labels.txt"):
    """Reads labels from file or uses defaults."""
    if not os.path.exists(path):
        return ["General Trash", "Construction Debris", "Organic", "Recyclable", "Hazardous"]
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# -------------------------
# Image Preprocessing
# -------------------------
def preprocess_image(image: Image.Image):
    """Resize to expected input shape and normalize."""
    if MODEL_READY and input_details is not None:
        h = int(input_details[0]["shape"][1])
        w = int(input_details[0]["shape"][2])
    else:
        h, w = 224, 224
    img = image.convert("RGB").resize((w, h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -------------------------
# Prediction logic
# -------------------------
def predict_from_image(pil_image):
    labels = load_labels()

    # If real model exists
    if MODEL_READY:
        input_data = preprocess_image(pil_image)

        # Handle quantization
        if input_details[0]["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] in (np.float32, np.float64):
            probs = tf.nn.softmax(output_data[0]).numpy()
        else:
            probs = output_data[0].astype(np.float32)
            probs = probs / probs.sum()
    else:
        # Fallback fake output
        probs = np.random.dirichlet(np.ones(len(labels)), size=1)[0]
        print("⚠️ Using simulated predictions (no model found).")

    pairs = list(zip(labels, probs))
    return sorted(pairs, key=lambda x: x[1], reverse=True)


# -------------------------
# Manual test
# -------------------------
if __name__ == "__main__":
    img_path = "sample.jpg"
    if os.path.exists(img_path):
        result = predict_from_image(Image.open(img_path))
        print(result)
    else:
        print("No sample image found, but code runs fine.")
