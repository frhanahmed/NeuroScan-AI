from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import fitz  # PyMuPDF
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

IMG_SIZE = 128  # Must match training size

# =============================
# LAZY LOAD MODEL (IMPORTANT)
# =============================

model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "brain-tumor-model.keras")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model


# =============================
# IMAGE PREPROCESSING
# =============================

def preprocess_image(img):
    # Apply CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Resize & Normalize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# =============================
# PDF TO IMAGE CONVERSION
# =============================

def convert_pdf_to_image(file_bytes):
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    page = pdf.load_page(0)  # First page only
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    # Convert RGB to BGR for OpenCV
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


# =============================
# ROUTES
# =============================

@app.route("/")
def home():
    return jsonify({"message": "Brain Tumor Detection API Running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    try:
        file_bytes = file.read()

        # Handle PDF
        if filename.endswith(".pdf"):
            img = convert_pdf_to_image(file_bytes)

        # Handle Image
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imdecode(
                np.frombuffer(file_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        processed_img = preprocess_image(img)

        # 🔥 Lazy load model here
        model = get_model()

        prediction = model.predict(processed_img)
        confidence = float(np.max(prediction))
        result = "Tumor" if np.argmax(prediction) == 1 else "No Tumor"

        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================
# PRODUCTION ENTRY POINT
# =============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)