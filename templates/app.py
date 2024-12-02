
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

IMAGE_SIZE = (128, 128)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):  
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASS_NAMES = [
    "BA- cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
    
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return "File type is not allowed. Only PNG, JPG, JPEG, and GIF are accepted.", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        img = Image.open(file_path)  # Open the image using PIL
        img = img.convert('RGB')  # Convert the image to RGB to handle all formats
        img = img.resize(IMAGE_SIZE)  # Resize the image
    except Exception as e:
        return f"Error processing image: {str(e)}", 400

    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_index]

    os.remove(file_path)

    return render_template('result.html', prediction=predicted_class)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
