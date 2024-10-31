
import librosa
import numpy as np
import pandas as pd
import os

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Mean of the MFCC features
    return mfcc_mean

# Example directory structure
music_genres = ['blues', 'classical', 'country', 'disco', 'hiphop' , 'hiphop', 'metal', 'pop', 'reggae' ,'jazz', 'rock']
dataset = []

# Loop through each genre folder
for genre in music_genres:
    genre_folder = f'C:/Users/mahdi/programing/Data/genres_original/{genre}/'
    
    # Loop through each audio file in the genre folder
    for filename in os.listdir(genre_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(genre_folder, filename)
            features = extract_features(file_path)
            features = np.append(features, genre)  # Add genre label
            dataset.append(features)

# Create a DataFrame
df = pd.DataFrame(dataset, columns=[f'MFCC_{i+1}' for i in range(13)] + ['Genre'])

# Save to CSV (optional)
df.to_csv('music_genre_dataset.csv', index=False)



