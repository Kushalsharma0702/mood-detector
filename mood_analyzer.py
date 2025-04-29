import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib
import os

def record_audio(duration=5, filename="output.wav"):
    fs = 44100
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Saved:", filename)
    return filename

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_mood(features, model_path="emotion_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found.")
    model = joblib.load(model_path)
    return model.predict([features])[0]

def main():
    audio_file = record_audio()
    features = extract_features(audio_file)
    mood = predict_mood(features)
    print("🎧 Detected Mood:", mood)

if __name__ == "__main__":
    main()
