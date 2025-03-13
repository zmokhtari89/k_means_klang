import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

#defining a function to extract features form audio files
def extract_features(file_path):
    ''' using librosa to load the audio files(s) and extract key features to later be stored in a dataframe'''
    # Load and trim the audio file
    y, sr = librosa.load(file_path)
    audio_file, _ = librosa.effects.trim(y)
    
    # Length (in samples)
    length = audio_file.shape[0]
    
    # Zero Crossing Rate
    zero_crossings = librosa.zero_crossings(audio_file, pad=False)
    zero_crossings_rate_mean = np.mean(zero_crossings)
    zero_crossings_rate_var = np.var(zero_crossings)
    
    # Harmonics & Percussive Components (HPSS)
    y_harm, y_perc = librosa.effects.hpss(audio_file)
    harmony_mean = np.mean(y_harm)
    harmony_var = np.var(y_harm)
    perceptr_mean = np.mean(y_perc)
    perceptr_var = np.var(y_perc)
    
    # Tempo
    tempo_value, _ = librosa.beat.beat_track(y=audio_file, sr=sr)
    tempo = tempo_value.item()
    
    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroids)
    spectral_centroid_var = np.var(spectral_centroids)
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_var = np.var(spectral_rolloff)
    
    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_file, sr=sr)
    spectral_bandwidth_mean = np.mean(bandwidth)
    spectral_bandwidth_var = np.var(bandwidth)
    
    # Chroma Frequencies
    hop_length = 5000  # Adjust for granularity
    chromagram = librosa.feature.chroma_stft(y=audio_file, sr=sr, hop_length=hop_length)
    chroma_mean = np.mean(chromagram)
    chroma_var = np.var(chromagram)
    
    # RMS Energy
    rms_values = librosa.feature.rms(y=audio_file)
    rms_mean = np.mean(rms_values)
    rms_var = np.var(rms_values)
    
    # MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)
    
    # Build feature dictionary
    features = {
        'file_name': os.path.basename(file_path),
        'length_samples': length,
        'zero_crossings_rate_mean': zero_crossings_rate_mean,
        'zero_crossings_rate_var': zero_crossings_rate_var,
        'harmony_mean': harmony_mean,
        'harmony_var': harmony_var,
        'perceptr_mean': perceptr_mean,
        'perceptr_var': perceptr_var,
        'tempo': tempo,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_var': spectral_centroid_var,
        'rolloff_mean': rolloff_mean,
        'rolloff_var': rolloff_var,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'spectral_bandwidth_var': spectral_bandwidth_var,
        'chroma_mean': chroma_mean,
        'chroma_var': chroma_var,
        'rms_mean': rms_mean,
        'rms_var': rms_var
    }
    
    # Add MFCC features (20 coefficients)
    for i in range(len(mfcc_means)):
        features[f'mfcc_mean_{i+1}'] = mfcc_means[i]
        features[f'mfcc_var_{i+1}'] = mfcc_vars[i]
    
    return features

# Define main folder path containing subfolders with audio files
main_folder_path = '../../raw_data/Data/genres_original'

# Collect file paths from all subfolders using os.walk()
file_paths = []
for root, dirs, files in os.walk(main_folder_path):
    for filename in files:
        if filename.lower().endswith(('.wav', '.mp3', '.flac')):
            file_paths.append(os.path.join(root, filename))

# Use joblib to process files in parallel
data_list = Parallel(n_jobs=-1)(delayed(extract_features)(fp) for fp in file_paths)

# Create a DataFrame from the list of feature dictionaries
df= pd.DataFrame(data_list)