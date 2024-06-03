from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the model
model_path = 'E:/classroomProject/classroom_detection.h5'
model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image size parameter
image_size = (128, 128)

# Define class labels (update with your actual class labels)
class_labels = ['Not in classroom', 'You are present in classroom']  # Update this with your actual class labels

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = load_and_preprocess_image(filepath)
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        class_label = class_labels[class_idx]
        return render_template('result.html', prediction=class_label)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
