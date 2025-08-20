import io
import re
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFile
import tensorflow as tf
import warnings

# Pillow settings to prevent warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

app = Flask(__name__, template_folder="templates")

# Load the trained CNN model
model = tf.keras.models.load_model("digit_cnn.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the base64 image data from POST request
        data = request.get_json()
        img_data = re.sub("^data:image/.+;base64,", "", data["image"])
        
        # Open the image, convert to grayscale
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L")
        
        # Resize to 28x28 (MNIST input size)
        img = img.resize((28, 28))
        
        # Convert to numpy array
        img_arr = np.array(img)
        
        # Invert colors (MNIST has white digit on black)
        img_arr = 255 - img_arr
        
        # Normalize to 0-1
        img_arr = img_arr / 255.0
        
        # Add batch and channel dimensions (1,28,28,1)
        img_arr = img_arr.reshape(1, 28, 28, 1)
        
        # Predict probabilities for each digit
        preds = model.predict(img_arr)
        prediction = int(np.argmax(preds))
        probabilities = preds.tolist()[0]  # full probability array
        
        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
