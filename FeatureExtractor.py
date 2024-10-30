import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of the audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Take mean to get a fixed-size array
    return mfcc_mean


result = extract_features('C:/Users/mahdi/programing/musicOrganizer/Data/genres_original/blues/blues.00000.wav')
print(result)