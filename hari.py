# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:13:16 2024

@author: rashm
"""

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import os

app = Flask(__name__)
model = MobileNetV2(weights='imagenet')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        img, img_array = load_and_preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_classes = decode_predictions(predictions, top=3)[0]

        result = [{'label': label, 'score': float(score)} for (_, label, score) in predicted_classes]
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
