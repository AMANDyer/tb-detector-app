# main.py - FastAPI for TB Detection
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Create the app
app = FastAPI(
    title="TB X-ray Detector API",
    description="Upload a chest X-ray → Get AI prediction (Normal or Tuberculosis)",
    version="1.0"
)

# Load your trained model (only once when server starts)
print("Loading model...")
model = tf.keras.models.load_model("my_model.keras")
print("Model loaded successfully!")

# Preprocessing function (same as training)
def preprocess_image(image: Image.Image):
    img = image.resize((300, 300))  # Your IMG_SIZE was 300
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # MobileNetV2 preprocessing
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_array = preprocess_input(img_array)
    return img_array

# Root endpoint - welcome message
@app.get("/")
def home():
    return {"message": "Welcome to TB Detector API! Go to /docs to test."}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if file is image
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "File is not an image"})
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess and predict
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0][0]  # Single value
    
    # Interpret result
    if prediction < 0.5:
        result = "NORMAL"
        confidence = float(1 - prediction)
    else:
        result = "TUBERCULOSIS"
        confidence = float(prediction)
    
    return {
        "result": result,
        "confidence": round(confidence, 4),
        "message": f"{result} with {confidence:.2%} confidence",
        "note": "This is an AI prediction. Always consult a doctor."
    }