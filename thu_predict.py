import librosa , librosa.display
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import math
import os

DATASET_PATH ='GTZAN/Data/genres_original/'
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION #so that is the number of amplitude value to represent each file.wav

def save_mfcc(dataset_path  , n_mfcc = 13 , n_fft = 2048 , hop_length = 512 , num_segments = 5):
    data = {
        'mapping' :[] ,
        'mfcc' :[] ,
        'labels' : []
    }
    num_sample_per_segment = int(SAMPLES_PER_TRACK / num_segments) #so here every 30 second we would split then into 5 segments
    expected_num_mfcc_vector_per_segment = math.ceil(num_sample_per_segment / hop_length)
    print(f'{expected_num_mfcc_vector_per_segment} that is the length of the sequence')
    
    
    for i , (dirpath , dirnames , filenames) in enumerate(os.walk(dataset_path)):
        
        
        if dirpath != dataset_path:
            semantic_label = dirpath.split('/')[-1]
            data['mapping'].append(semantic_label)
            for file in filenames:
                file_path = os.path.join(dirpath , file)
                try :
                    signal , sr = librosa.load(file_path , sr = SAMPLE_RATE)
                    for s in range(num_segments):
                        start_sample = num_sample_per_segment * s
                        finish_sample = start_sample + num_sample_per_segment
                        mfcc = librosa.feature.mfcc(y = signal[start_sample:finish_sample] , sr = SAMPLE_RATE , 
                                                   n_mfcc=13 , n_fft = n_fft , hop_length = hop_length)
                        mfcc = mfcc.T # as we have the data (sequence number , number of features)
                        if len(mfcc) == expected_num_mfcc_vector_per_segment:
                            data['mfcc'].append(mfcc.tolist())
                            data['labels'].append(i-1)
                except:
                    pass
                        
            print(f"{dirpath.split('/')[-1]} is loaded successfully")
                        
    return data

data_dict = save_mfcc(DATASET_PATH  , num_segments=10)

print(data_dict)