import os
import numpy as np
import librosa
import joblib

np.random.seed(1)

fs = 44100
n_mels = 128
n_mfcc = 13

def extract_features(file, fs=44100, n_mels=128, n_mfcc=13):
    y, sr = librosa.load(file, sr=fs)
    y /= np.max(np.abs(y))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

model_dict = joblib.load('music.joblib')
knn_classifier = model_dict['model']
scaler = model_dict['scaler']
label_encoder = model_dict['label_encoder']

new_audio_file = r"C:\Users\JaySs\OneDrive\Desktop\New folder (2)\flute.wav"
new_feature_vector = extract_features(new_audio_file)
scaled_new_feature_vector = scaler.transform([new_feature_vector])
predicted_label_num = knn_classifier.predict(scaled_new_feature_vector)
predicted_label_name = label_encoder.inverse_transform(predicted_label_num)[0]

print(f"Predicted class for {os.path.basename(new_audio_file)}: {predicted_label_name}")
