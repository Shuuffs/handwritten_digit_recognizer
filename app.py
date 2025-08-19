from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np, cv2, base64

app = Flask(__name__)
model = load_model("digit_cnn.h5")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert colors (canvas: black on white → MNIST: white on black)
    img = cv2.bitwise_not(img)

    # Normalize to 0-1
    img = img.astype("float32") / 255.0

    # Reshape to (1, 28, 28, 1)
    img = np.expand_dims(img, axis=(0, -1))

    # Prediction
    prediction = model.predict(img)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({"prediction": predicted_digit})


if __name__ == "__main__":
    app.run(debug=True)
