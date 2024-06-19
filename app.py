from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import Counter
import os
app = Flask(__name__)

# Load the Keras model
model = load_model('luk_cnn_model.keras')

# Define the input shape for preprocessing
input_shape = (224, 224, 3)

# Define preprocessing function for the image
def preprocess_image(image):
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = image / 255.0  # Normalize pixel values
    return image

# Define classes based on the integer values
classes = ['Malignant Early', 'Malignant Pre', 'Malignant Pro']

# Define a route to render the HTML upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Define a route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    img_file = request.files['image']

    # Read the image file
    original_img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    img = preprocess_image(original_img)

    # Make prediction using the loaded model
    predictions = model.predict(np.expand_dims(img, axis=0))

    # Convert the first column into integer type for each prediction
    predictions = predictions.astype(int)

    # Extract the first column values
    first_column_values = predictions[:, :, 0].flatten()

    # Determine the most frequent value in the first column
    most_common_value, _ = Counter(first_column_values).most_common(1)[0]

    # Get the type corresponding to the most common value
    result_type = classes[most_common_value]
    
   # Save the original input image to a temporary file within the static folder
    temp_image_path = os.path.join('static', 'temp_image.jpg')
    cv2.imwrite(temp_image_path, original_img)

    # Get the URL for the temporary image file
    temp_image_url = url_for('static', filename='temp_image.jpg')

    
    
    # Return the result to the HTML page
    return render_template('result.html',  image=temp_image_url, result=result_type)

if __name__ == '__main__':
    app.run(debug=True)
