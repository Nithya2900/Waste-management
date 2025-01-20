from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
import traceback

# Load the trained model
model = load_model('garbage_classifier_model.h5')

# Create a Flask app
app = Flask(__name__)

# Predefined class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Serve static files from the public folder
@app.route('/public/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'public'), filename)

# Route to serve the homepage (mainwebpage.html)
@app.route('/')
def index():
    return render_template('mainwebpage.html')

# Route to serve the recycle.html page (from the public folder)
@app.route('/recycle')
def recycle():
    return send_from_directory(os.path.join(app.root_path, 'public'), 'recycle.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is part of the POST request
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected for uploading", 400
        
        if file:
            # Open the image using PIL
            img = Image.open(file)
            
            # Convert image to RGB if it has an alpha channel
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize the image to match the input shape required by the model
            img = img.resize((224, 224))
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Scale pixel values to [0, 1]
            img_array = img_array / 255.0
            
            # Add a batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make a prediction
            prediction = model.predict(img_array)
            
            # Get the class with the highest probability
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_names[predicted_class_index]
            
            return f"Predicted Class: {predicted_class}"
        else:
            return "Invalid file", 400
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()  # Get detailed error traceback
        print(f"Error occurred: {error_trace}")
        return "Error occurred while processing the image.", 500

if __name__ == '__main__':
    app.run(debug=True)
