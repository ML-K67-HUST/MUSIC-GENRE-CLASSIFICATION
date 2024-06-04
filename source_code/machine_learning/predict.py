import pandas as pd
import numpy as np 
import pickle

import librosa
import librosa.display
import warnings
import joblib

from keras.models import load_model # type: ignore
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
        'harmony_mean': [harmony_mean],
        'harmony_var': [harmony_var],
        'perceptr_mean': [perceptr_mean],
        'perceptr_var': [perceptr_var],
        'tempo' :[tempo]
    })
    
    # Add MFCC features
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = [mfcc_means[i-1]]
        features[f'mfcc{i}_var'] = [mfcc_vars[i-1]]
    return features
current_dir = os.path.dirname(os.path.abspath(__file__))

def analyze_audio(audio_file):
    y, sr = librosa.load(audio_file)
    l = len(y)//2
    features_comb = []
    start = 0
    while start + 30*sr < len(y):
        scaler = joblib.load(current_dir.replace('machine_learning','') +'saved_model/scaler.pkl')
        feature = scaler.transform(np.array(extract_features(y[start:start+30*sr],sr)))
        features_comb.append(feature)
        start = start + 30*sr
    return features_comb
def predict_(aud):
    features_comb = analyze_audio(aud)  # Assuming analyze_audio extracts features

    # Load pre-trained models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(current_dir.replace('machine_learning','') +'saved_model/stack_model.pkl', 'rb') as file:
        stack = pickle.load(file)
    with open(current_dir.replace('machine_learning','') +'saved_model/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    with open(current_dir.replace('machine_learning','') +'saved_model/svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    nn_model = load_model(current_dir.replace('machine_learning','') +'saved_model/nn_model.keras')  # Assuming load_model loads the neural network

    # Make predictions and store them in a list of dictionaries
    predictions, predictions_stack = [], []

    dic_knn = {"blues":0,"classical":0,"country":0,"disco":0,"hiphop":0,"jazz":0,"metal":0,"pop":0,"reggae":0,"rock":0}
    dic_svm = {"blues":0,"classical":0,"country":0,"disco":0,"hiphop":0,"jazz":0,"metal":0,"pop":0,"reggae":0,"rock":0}
    dic_nn = {"blues":0,"classical":0,"country":0,"disco":0,"hiphop":0,"jazz":0,"metal":0,"pop":0,"reggae":0,"rock":0}
    dic_stack = {"blues":0,"classical":0,"country":0,"disco":0,"hiphop":0,"jazz":0,"metal":0,"pop":0,"reggae":0,"rock":0}

    for feature in features_comb:
        knn_pred = genres[knn_model.predict(feature)[0]]
        svm_pred = genres[svm_model.predict(feature)[0]]
        nn_pred = genres[nn_model.predict(feature)[0].argmax(axis=-1)]

        dic_knn[knn_pred] = dic_knn.get(knn_pred, 0) + 1
        dic_svm[svm_pred] = dic_svm.get(svm_pred, 0) + 1
        dic_nn[nn_pred] = dic_nn.get(nn_pred, 0) + 1

    stack_pred = stack.predict(np.array(features_comb).reshape(-1, 57))
    for p in stack_pred:
        dic_stack[genres[p]] = dic_stack.get(genres[p], 0) + 1

    dic_knn_p = {x: str(round(dic_knn[x] * 100 / sum(dic_knn.values()))) + '%' for x in dic_knn}
    dic_svm_p = {x: str(round(dic_svm[x] * 100 / sum(dic_svm.values()))) + '%' for x in dic_svm}
    dic_nn_p = {x: str(round(dic_nn[x] * 100 / sum(dic_nn.values()))) + '%' for x in dic_nn}
    dic_stack_p = {x: str(round(dic_stack[x] * 100 / sum(dic_stack.values()))) + '%' for x in dic_stack}

    max_confidence_genre = max(dic_stack, key=dic_stack.get)
    max_confidence = round(dic_stack[max_confidence_genre] * 100 / sum(dic_stack.values()))

    sorted_genres = sorted(dic_stack, key=dic_stack.get, reverse=True)
    first_genre, second_genre, third_genre, fourth_genre = sorted_genres[:4]
    max_confidence = round(dic_stack[first_genre] * 100 / sum(dic_stack.values()))
    second_confidence = round(dic_stack[second_genre] * 100 / sum(dic_stack.values()))


    if max_confidence == 100:
        confidence_message = f"Absolutely {first_genre}!"
    elif max_confidence >= 80:
        confidence_message = f"I'm pretty sure this is {first_genre}"
    elif max_confidence >= 70:
        confidence_message = f"I'm pretty sure this is {first_genre}, but I feel a little bit of {second_genre} inside the song"
    elif max_confidence >= 50:
        if second_confidence >= 30:
            confidence_message = f"{first_genre}? Kinda {first_genre} mixed with some {second_genre}"
        else:
            confidence_message = f"Hmm, I think this is {first_genre} mixed with a little bit of {second_genre}"
    elif max_confidence >= 30:
        if second_confidence >= 30:
            confidence_message = f"I'm a little bit confused... this can be a {first_genre} song mixed with {second_genre}, I also find {third_genre}'s melody here"
        else:
            confidence_message = f"It's hard to decide the specific genre of this... it can be {first_genre} mixed with a little bit {second_genre} and {third_genre}"
    elif max_confidence < 30:
        if max_confidence < 10:
            confidence_message = f"I'm not confident at all, this song can be {first_genre} mixed with {second_genre}, I also see {third_genre} inside, and even {fourth_genre}..."
        else:
            confidence_message = f"Ugh...I think this is {first_genre} mixed with {second_genre} and some {third_genre}... It's quite strange to me..."


    predictions.append({'model': 'K-Nearest Neighbors', 'genre': dic_knn_p})
    predictions.append({'model': 'Support Vector Machine', 'genre': dic_svm_p})
    predictions.append({'model': 'Neural Network', 'genre': dic_nn_p})
    predictions_stack.append({'model': 'Stacking Ensemble Learning', 'genre': dic_stack_p})

    return predictions, predictions_stack, max_confidence, confidence_message
