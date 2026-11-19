# Mood Detector: Speech Emotion Recognition

Python project that listens to your voice and predicts your mood from audio features.

## Done so far

- Model training script (`kaggle.py`): loads the labelled speech dataset in `audio_speech_actors_01-24/`, extracts audio features with librosa and trains a scikit-learn classifier saved as `emotion_model.pkl`
- Command-line analyzer (`mood_analyzer.py`): records from the microphone (sounddevice), extracts features and predicts the mood
- Streamlit app (`stream.py`): live recording and mood prediction with a timestamped history

## Stack

Python, librosa, scikit-learn, joblib, sounddevice, SciPy, Streamlit

## Run locally

```bash
pip install -r requirements.txt
python kaggle.py          # optional: retrain the model
python mood_analyzer.py   # command-line mode
streamlit run stream.py   # web app
```

## Future scope

- Report accuracy, confusion matrix and per-emotion results for the trained model
- Try deep models (CNN or wav2vec features) and data augmentation
- Real-time streaming prediction with smoothing
- Move the dataset (about 225 MB) out of the repo and document where to download it
- Add the speaking-style feedback from [Voice_analysis](https://github.com/Kushalsharma0702/Voice_analysis)
