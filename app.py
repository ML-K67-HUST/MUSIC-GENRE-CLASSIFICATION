import pandas as pd
import numpy as np 
import pickle

import librosa
import librosa.display
import warnings
import joblib

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore')  # Suppress other warnings

# Now import TensorFlow and Keras modules
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 

seed = 42
np.random.seed(seed)

hop_length = 512

n_fft = 2048

genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# Function to extract features from audio file

def extract_features(y, sr):
    # Extract features
    chroma_stft_mean = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).mean()
    chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).var()
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).var()
    harmony, perceptr = librosa.effects.hpss(y)
    harmony_mean = harmony.mean()
    harmony_var = harmony.var()
    perceptr_mean = perceptr.mean()
    perceptr_var = perceptr.var()

    tempo = librosa.beat.beat_track(y=y, sr=sr, units='time')[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = mfccs.mean(axis=1)
    mfcc_vars = mfccs.var(axis=1)
    
    # Create DataFrame
    features = pd.DataFrame({
        'chroma_stft_mean': [chroma_stft_mean],
        'chroma_stft_var': [chroma_stft_var],
        'rms_mean': [rms_mean],
        'rms_var': [rms_var],
        'spectral_centroid_mean': [spectral_centroid_mean],
        'spectral_centroid_var': [spectral_centroid_var],
        'spectral_bandwidth_mean': [spectral_bandwidth_mean],
        'spectral_bandwidth_var': [spectral_bandwidth_var],
        'rolloff_mean': [rolloff_mean],
        'rolloff_var': [rolloff_var],
        'zero_crossing_rate_mean': [zero_crossing_rate_mean],
        'zero_crossing_rate_var': [zero_crossing_rate_var],
        'harmony_mean': [harmony_mean.mean()],
        'harmony_var': [harmony_var.var()],
        'perceptr_mean': [perceptr_mean.mean()],
        'perceptr_var': [perceptr_var.var()],
        'tempo' :[tempo]
    })
    
    # Add MFCC features
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = [mfcc_means[i-1]]
        features[f'mfcc{i}_var'] = [mfcc_vars[i-1]]
    return features

def analyze_audio(audio_file):
    y, sr = librosa.load(audio_file)
    l = len(y)//2
    features_comb = []
    start = 0
    while start + 30*sr < len(y):
        scaler = joblib.load('scaler.pkl')
        feature = scaler.transform(np.array(extract_features(y[start:start+30*sr],sr)))
        features_comb.append(feature)
        start = start + 30*sr
    return features_comb
def predict_(aud):
    features_comb = analyze_audio(aud)  # Assuming analyze_audio extracts features

    # Load pre-trained models
    with open('saved_model/ens_model.pkl', 'rb') as file:
        ens_model = pickle.load(file)
    with open('saved_model/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    with open('saved_model/svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    nn_model = load_model('saved_model/nn_model.keras')  # Assuming load_model loads the neural network

    # Make predictions and store them in a list of dictionaries
    predictions = []
    for feature in features_comb:
        model_predictions = [
            {"model": "KNN", "genre": genres[knn_model.predict(feature)[0]]},
            {"model": "Ensemble", "genre": genres[ens_model.predict(feature)[0]]},
            {"model": "SVM", "genre": genres[svm_model.predict(feature)[0]]},
            {"model": "Neural Net", "genre": genres[nn_model.predict(feature)[0].argmax(axis=-1)]}
        ]
        predictions.extend(model_predictions)

    return predictions


# audio = '/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/genres_original/disco/disco.00000.wav'
# print(predict_(audio))

from flask import Flask, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings

app = Flask(__name__, template_folder='/home/khangpt/MUSIC-GEN-PROJ')
# Set the upload folder path (replace with your actual path)
app.config['UPLOAD_FOLDER'] = '/home/khangpt/MUSIC-GEN-PROJ/user_song'
app.config['SECRET_KEY'] = '123'  # Required for using sessions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Store filename in session for prediction route (alternative approaches possible)
        session['uploaded_filename'] = filename
        return jsonify({'message': 'Song uploaded successfully!'})

    return jsonify({'error': 'Failed to upload file'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve uploaded filename (assuming it's stored in session)
    filename = session.get('uploaded_filename')

    # Alternative: retrieve filename from request object (if not using session)
    # if not filename:
    #     filename = request.args.get('filename')  # Assuming filename passed as query param

    if not filename:
        return jsonify({'error': 'Missing uploaded filename'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predictions = predict_(file_path)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)


