import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
import tempfile
import parselmouth
from parselmouth.praat import call
import joblib
from tensorflow.keras.models import load_model

# Load models
speech_model = load_model(r"C:\Users\medhi\OneDrive\Desktop\autism_detection\speech_model\speech_autism_model.h5")
behavior_model = joblib.load(r"C:\Users\medhi\OneDrive\Desktop\autism_detection\behavioral_model\behavior_model.pkl")
scaler = joblib.load(r"C:\Users\medhi\OneDrive\Desktop\autism_detection\speech_model\scaler.pkl")

SR = 22050
MAX_LEN = 110250

st.set_page_config(page_title="Autism Detection System", layout="centered")
st.title("🧠 Autism Detection System")

# --- Global variable for recorded audio path
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None


# ------------------ Feature Extraction ------------------
def extract_speech_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SR)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ])
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        features = np.zeros(65)

    try:
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        pitch_feats = [mean_pitch, jitter, shimmer]
    except Exception as e:
        st.warning(f"Pitch extraction failed: {e}")
        pitch_feats = [0, 0, 0]

    full_feature = np.concatenate([features, pitch_feats]).reshape(1, -1)
    full_feature = scaler.transform(full_feature)
    return full_feature


# ------------------ Record Audio ------------------
def record_audio(duration=5, sr=SR):
    st.info("🎤 Recording for 5 seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio, sr)
    st.success("✅ Recording complete!")
    return tmp_file.name


# ------------------ Form UI ------------------
st.header("Step 1: Behavioral Questionnaire")
st.markdown("Please answer the following questions about the child.")

questions = [
    "Does your child respond to their name?",
    "Does your child make eye contact?",
    "Does your child enjoy social interactions?",
    "Does your child point to objects of interest?",
    "Does your child use simple gestures (e.g., waving)?",
    "Does your child imitate others?",
    "Does your child play with toys appropriately?",
    "Does your child show interest in peers?",
    "Does your child use age-appropriate speech?",
    "Does your child understand simple instructions?"
]

answers = []
for q in questions:
    answers.append(st.checkbox(q))

# Upload or record section
st.header("Step 2: Speech Test")
upload_file = st.file_uploader("Upload child's voice recording (.wav)", type=["wav"])

if st.button("🎙️ Record for 5 Seconds"):
    path = record_audio()
    st.session_state.audio_path = path
    audio_bytes = open(path, "rb").read()
    st.audio(audio_bytes, format="audio/wav")

# -------------- Submit Button --------------
if st.button("🔍 Predict Autism Risk"):
    if all(isinstance(ans, bool) for ans in answers):
        behavior_input = np.array([[int(a) for a in answers]])
        behavior_result = behavior_model.predict(behavior_input)[0]

        # Determine audio source
        final_audio_path = None
        if upload_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(upload_file.read())
                final_audio_path = tmp.name
        elif st.session_state.audio_path:
            final_audio_path = st.session_state.audio_path

        if final_audio_path:
            speech_input = extract_speech_features(final_audio_path)
            speech_prob = speech_model.predict(speech_input)[0][0]
            speech_result = "Autistic" if speech_prob > 0.5 else "Non-Autistic"
        else:
            st.error("❌ No speech input found. Please upload or record.")
            speech_result = "Unavailable"
            speech_prob = 0.0

        # Show results
        st.subheader("🧾 Results")
        st.success(f"Behavioral Model: {'Autistic' if behavior_result == 0 else 'Non-Autistic'}")
        st.success(f"Speech Model Prediction: {speech_result} ({speech_prob:.2f})")
    else:
        st.warning("⚠️ Please answer all behavioral questions.")
