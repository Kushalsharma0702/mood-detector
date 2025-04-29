import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Map emotion codes to labels
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def load_data(data_dir):
    X, y = [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                if emotion_code in emotion_map:
                    label = emotion_map[emotion_code]
                    features = extract_features(os.path.join(root, file))
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

def train_model():
    print("[INFO] Loading data...")
    X, y = load_data("audio_speech_actors_01-24")
    
    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[INFO] Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "emotion_model.pkl")
    print("[✅] Model saved as emotion_model.pkl")

if __name__ == "__main__":
    train_model()
