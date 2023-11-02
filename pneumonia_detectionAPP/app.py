import os
import numpy as np
import librosa
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Define the directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('models/MY_Model.h5')  # Replace with your model's filename

# Constants for feature extraction
SAMPLE_RATE = 22050
DURATION = 4
MFCC_FEATURES = 2  # Change this to match the model's input shape

# Function to extract MFCC features from an audio clip
def extract_features(file_path, target_length):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)

    # Pad or truncate to the target length
    if mfccs.shape[1] < target_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > target_length:
        mfccs = mfccs[:, :target_length]

    return mfccs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if audio_file:
        # Use the secure_filename function to ensure safe filenames
        filename = secure_filename(audio_file.filename)

        # Save the uploaded audio file to the uploads directory
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        # Extract MFCC features from the audio clip
        target_length = 20  # The target length for your model's input shape
        audio_features = extract_features(audio_path, target_length)

        # Reshape the audio features to match the model's input shape
        audio_features = audio_features.reshape(1, 20, MFCC_FEATURES)  # Ensure MFCC_FEATURES is set correctly

        # Make predictions using your model
        predictions = model.predict(audio_features)

        # Assuming binary classification (Pneumonia and Normal)
        if predictions[0][0] > predictions[0][1]:
            predicted_class = 'Normal'
        else:
            predicted_class = 'Pneumonia'

        return jsonify({'prediction': predicted_class})

    return jsonify({'error': 'Failed to make a prediction'})

if __name__ == '__main__':
    app.run(debug=True)
