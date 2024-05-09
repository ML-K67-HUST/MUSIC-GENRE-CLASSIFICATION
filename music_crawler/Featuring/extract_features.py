import librosa 
import os
import pandas as pd

HOP_LENGTH = 512

def extract_features(y, sr):
    # Extract features
    chroma_stft_mean = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH).mean()
    chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH).var()
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH).var()
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

def analyze_audio(audio_file):
    y, sr = librosa.load(audio_file)
    l = len(y)//2
    features_comb = []
    start = 0
    while start + 30*sr < len(y):
        features_comb.append((extract_features(y[start:start+30*sr],sr),start))
        start = start + 30*sr
    return features_comb

def playlist_to_features_csv(directory,genre):
    # Initialize an empty DataFrame
    df = pd.DataFrame()
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            filepath = os.path.join(directory, filename)
            for data, start in analyze_audio(filepath):
                df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    df['label'] = genre
    # Save the csv file to Music Crawler/CSV/
    # df.to_csv('/content/CSV/crawled_'+genre+'_dataframe.csv')
    return df