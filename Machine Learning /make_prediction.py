import pandas as pd
import numpy as np 
import pickle

import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

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
        features_comb.append((extract_features(y[start:start+30*sr],sr),start))
        start = start + 30*sr
    return features_comb
def predict_(aud):
    dic_knn, dic_new_knn, dic_ens, dic_svm = {},{},{},{}
    features_comb = analyze_audio(aud)
    with open('ens_model.pkl','rb') as file:
        ens_model = pickle.load(file)
    with open('knn_model.pkl','rb') as file:
        knn_model = pickle.load(file)
    with open('svm_model.pkl','rb') as file:
        svm_model = pickle.load(file)
    with open('new_knn_model.pkl','rb') as file:
        new_knn = pickle.load(file)
    for c,r in features_comb:
        pred_knn = knn_model.predict(c)
        dic_knn[genres[pred_knn[0]]] = dic_knn.get(pred_knn[0],0) + 1

        pred_new_knn = new_knn.predict(c)
        dic_new_knn[pred_new_knn[0]] = dic_new_knn.get(pred_new_knn[0],0) + 1

        pred_ens = ens_model.predict(c)
        dic_ens[pred_ens[0]] = dic_ens.get(pred_ens[0],0) + 1

        pred_svm = svm_model.predict(c)
        dic_svm[genres[pred_svm[0]]] = dic_svm.get(genres[pred_svm[0]],0) + 1

    print("KNN's prediction : ",{x:str(round(dic_knn[x]*100/sum(list(dic_knn.values())))) + '%' for x in dic_knn})
    print("New KNN's prediction : ",{x:str(round(dic_new_knn[x]*100/sum(list(dic_new_knn.values())))) + '%' for x in dic_new_knn})
    print("Ensemble method's prediction : ",{x:str(round(dic_ens[x]*100/sum(list(dic_ens.values())))) + '%' for x in dic_ens})
    print("SVM's Prediction : ",{x:str(round(dic_svm[x]*100/sum(list(dic_svm.values())))) + '%' for x in dic_svm})
    return 'Finished!'

# import matplotlib.pyplot as plt
# import numpy as np

# # Define genre colors
# genre_colors = {
#     "blues": "blue",
#     "classical": "green",
#     "country": "red",
#     "disco": "cyan",
#     "hiphop": "magenta",
#     "jazz": "yellow",
#     "metal": "orange",
#     "pop": "purple",
#     "reggae": "brown",
#     "rock": "gray"
# }
# genre_labels = list(genre_colors.keys())

# # Function to predict and visualize the distribution of genres
# def predict_and_visualize(aud):
#     features_comb = analyze_audio(aud)
#     for (c, r) in features_comb:
#         pred = KNN_predict(c)
#         genre_color = genre_colors.get(pred, "black")
#         start_time = r   # Start time of the segment in seconds
#         plt.scatter(start_time, 0, color=genre_color, marker='o', s=100)
    
#     # Plot settings
#     handles = [plt.scatter([], [], color=color, label=genre) for genre, color in genre_colors.items()]
#     plt.legend(handles, genre_labels, loc='upper right')

#     plt.show()

# Call the function to predict and visualize
audio = 'User Data Test Songs/jazzt.mp3'
print(predict_(audio))