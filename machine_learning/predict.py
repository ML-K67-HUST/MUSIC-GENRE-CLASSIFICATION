import pandas as pd
import numpy as np 
import pickle

import librosa
import librosa.display
import warnings
import joblib

from keras.models import load_model # type: ignore
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
    with open(current_dir.replace('machine_learning','') +'saved_model/ens_model.pkl', 'rb') as file:
        ens_model = pickle.load(file)
    with open(current_dir.replace('machine_learning','') +'saved_model/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    with open(current_dir.replace('machine_learning','') +'saved_model/svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    nn_model = load_model(current_dir.replace('machine_learning','') +'saved_model/nn_model.keras')  # Assuming load_model loads the neural network

    # Make predictions and store them in a list of dictionaries
    predictions = []
    
    dic_knn, dic_ens, dic_svm, dic_nn = {}, {}, {}, {}

    for feature in features_comb:

        knn_pred = genres[knn_model.predict(feature)[0]]
        ens_pred = genres[ens_model.predict(feature)[0]]
        svm_pred = genres[svm_model.predict(feature)[0]]
        nn_pred = genres[nn_model.predict(feature)[0].argmax(axis=-1)]

        dic_knn[knn_pred] = dic_knn.get(knn_pred,0) + 1
        dic_ens[ens_pred] = dic_ens.get(ens_pred,0) + 1
        dic_svm[svm_pred] = dic_svm.get(svm_pred,0) + 1
        dic_nn[nn_pred] = dic_nn.get(nn_pred,0) + 1
    dic_knn = {x:str(round(dic_knn[x]*100/sum(dic_knn.values()))) + '%' for x in dic_knn}
    dic_ens = {x:str(round(dic_ens[x]*100/sum(dic_ens.values()))) + '%' for x in dic_ens}
    dic_svm = {x:str(round(dic_svm[x]*100/sum(dic_svm.values()))) + '%' for x in dic_svm}
    dic_nn = {x:str(round(dic_nn[x]*100/sum(dic_nn.values()))) + '%' for x in dic_nn}

    predictions.append({'model':'KNN', 'genre':dic_knn})
    predictions.append({'model':'ENS', 'genre':dic_ens})
    predictions.append({'model':'SVM', 'genre':dic_svm})
    predictions.append({'model':'NN', 'genre':dic_nn})

    return predictions

