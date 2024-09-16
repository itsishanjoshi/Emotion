from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('-08.keras')  # Update with the correct model path

# Mapping emotion classes
EMOTION_CLASSES = ['anger', 'contempt', 'neutral', 'happy', 'surprise', 'fear', 'disgust', 'sad']

# Define a function to preprocess the image
def preprocess_image(image_path):
    try:
        # Load the image from the path
        image = cv2.imread(image_path)
        print(f"Loaded image shape: {image.shape}")  # Debug: Check if image loads correctly

        # Resize the image to 224x224 as expected by the model
        image = cv2.resize(image, (224, 224))

        # Convert image to RGB (in case it's grayscale or another format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize the image (convert pixel values from 0-255 to 0-1)
        image = image / 255.0

        # Expand dimensions to match the model input shape (1, 224, 224, 3)
        image = np.expand_dims(image, axis=0)
        print(f"Processed image shape: {image.shape}")  # Debug: Check image preprocessing output

        return image
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected for uploading'}), 400

        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess image and make prediction
            image_tensor = preprocess_image(file_path)
            if image_tensor is None:
                return jsonify({'error': 'Image preprocessing failed'}), 500  # Return error if preprocessing fails

            outputs = model.predict(image_tensor)  # Get model predictions
            print(f"Model Outputs: {outputs}")  # Debug: Check model prediction outputs

            # Assuming outputs are probabilities, find the predicted class
            predicted_class = np.argmax(outputs, axis=1)[0]
            print(f"Predicted Class: {predicted_class}")  # Debug: Check the predicted class index

            emotion = EMOTION_CLASSES[predicted_class]
            return jsonify({
                'filename': filename,
                'emotion': emotion
            })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

