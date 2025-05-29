import joblib
from flask import Flask, request, send_from_directory
import base64
import re
import numpy as np
#import tensorflow as tf
from PIL import Image
import io
import os

model_braille = joblib.load('braille.joblib')
app = Flask(__name__)

# Serve the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'braillify.htm')

# Receive image from frontend
@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    image_data = data['image']

    # Decode the image
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image_bytes = base64.b64decode(image_data)

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    array=np.array(image).flatten()
    return model_braille.predict([array])[0]
    # Convert to NumPy array, normalize, and prepare for TensorFlow
    #image_array = np.array(image) / 255.0
    #input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #input_tensor = tf.expand_dims(input_tensor, axis=0)
    #print(f"[INFO] Image received and ready for model. Shape: {input_tensor.shape}")
    return "Image processed and ready for model!"

if __name__ == '__main__':
    app.run(debug=True)