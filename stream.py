import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib
import streamlit as st
from datetime import datetime

# Set up temp folder for recordings
TEMP_FOLDER = "temp_recordings"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Audio recording
def record_audio(duration=5):
    fs = 44100
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(TEMP_FOLDER, f"recording_{timestamp}.wav")
    
    with st.spinner("Recording..."):
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write(filename, fs, audio)
    st.success(f"Saved recording as {filename}")
    return filename

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Load model and predict
def predict_mood(features, model_path="emotion_model.pkl"):
    if not os.path.exists(model_path):
        st.error("Model not found. Please train it first.")
        st.stop()
    model = joblib.load(model_path)
    return model.predict([features])[0]

# Streamlit App
st.set_page_config(page_title="Voice Mood Analyzer", layout="centered", initial_sidebar_state="expanded")
st.title("🎤 Voice-Based Mood Analyzer")

# Theme toggle
theme = st.sidebar.radio("Choose Theme", ("Light", "Dark"))

if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #111; color: white; }
        .stApp { background-color: #222; }
        </style>
    """, unsafe_allow_html=True)

st.markdown("Record your voice for 5 seconds and let the model analyze your mood.")

if st.button("Start Recording"):
    audio_path = record_audio()
    features = extract_features(audio_path)
    mood = predict_mood(features)
    st.markdown(f"### 🎧 Detected Mood: `{mood.capitalize()}`")

# Optional: list previous recordings
if st.checkbox("Show Previous Recordings"):
    recordings = sorted(os.listdir(TEMP_FOLDER))[-5:]
    for file in recordings:
        st.write(file)
