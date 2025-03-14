import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler


#----------------------------------------------------------------------------------------
#OPTION FOR FETCHING AND TRANSFORMING AUDIO FILE(S) FROM ONE FOLDER
#---------------------------------------------------------------------------------------

def extract_features(file_path):
    """ 
    Using librosa to load the audio file received from api, will transform data into key features, stored in a dataframe for preprocessing
    """
    
    #when in the .py we will ned to use os.join__file__ and then set the path 

    # Find audio file 
    general_path = '../../raw_data/Data'
    file_path = f'{general_path}/genres_original/jazz/jazz.00055.wav'
   
    #Load and trim audio file
    y, sr = librosa.load(str(file_path)) 
    audio_file, _ = librosa.effects.trim(y)

    #Extract features. When relevant, calculate mean and variance
    # Length (in samples)
    length = audio_file.shape[0]
    
    # Chroma Frequencies
    hop_length = 5000  # Adjust for granularity
    chromagram = librosa.feature.chroma_stft(y=audio_file, sr=sr, hop_length=hop_length)
    chroma_stft_mean = np.mean(chromagram)
    chroma_stft_var = np.var(chromagram)
    
    # RMS Energy
    rms_values = librosa.feature.rms(y=audio_file)
    rms_mean = np.mean(rms_values)
    rms_var = np.var(rms_values)
    
    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroids)
    spectral_centroid_var = np.var(spectral_centroids)
    
    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_file, sr=sr)
    spectral_bandwidth_mean = np.mean(bandwidth)
    spectral_bandwidth_var = np.var(bandwidth)
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_var = np.var(spectral_rolloff)
    
    # Zero Crossing Rate
    zero_crossings = librosa.zero_crossings(audio_file, pad=False)
    zero_crossing_rate_mean = np.mean(zero_crossings)
    zero_crossing_rate_var = np.var(zero_crossings)
    
    # Harmonics & Percussive Components (HPSS)
    y_harm, y_perc = librosa.effects.hpss(audio_file)
    harmony_mean = np.mean(y_harm)
    harmony_var = np.var(y_harm)
    perceptr_mean = np.mean(y_perc)
    perceptr_var = np.var(y_perc)
    
    # Tempo
    tempo_value, _ = librosa.beat.beat_track(y=audio_file, sr=sr)
    tempo = tempo_value.item()

    # MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)
    
    # Build feature dictionary
    #when in the .py we will ned to use os.join__file__ and then set the path 
    features = {
        'filename': os.path.basename(file_path),
        'length': length,
        'chroma_stft_mean': chroma_stft_mean,
        'chroma_stft_var': chroma_stft_var,
        'rms_mean': rms_mean,
        'rms_var': rms_var,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_var': spectral_centroid_var,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'spectral_bandwidth_var': spectral_bandwidth_var,
        'rolloff_mean': rolloff_mean,
        'rolloff_var': rolloff_var,
        'zero_crossing_rate_mean': zero_crossing_rate_mean,
        'zero_crossing_rate_var': zero_crossing_rate_var,
        'harmony_mean': harmony_mean,
        'harmony_var': harmony_var,
        'perceptr_mean': perceptr_mean,
        'perceptr_var': perceptr_var,
        'tempo': tempo,
    }
    
    # Add MFCC features (20 coefficients)
    for i in range(len(mfcc_means)):
        features[f'mfcc{i+1}_mean'] = mfcc_means[i]
        features[f'mfcc{i+1}_var'] = mfcc_vars[i]
        
    # Add the label column with a default value 'no_label'
    features['label'] = 'no_label'
    
    print("✅ data transformed into features")
    
    return features 



if __name__ == '__main__':
        extract_features()