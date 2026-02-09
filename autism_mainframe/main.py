import os
import sys
import time
import tempfile
import subprocess
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import librosa
import sounddevice as sd
import soundfile as sf
import tensorflow as tf

# =======================
# STREAMLIT SETUP
# =======================
st.set_page_config(page_title="Autism Risk Assessment", layout="centered")
st.title("🧠 Autism Risk Assessment Tool")

st.markdown("""
This tool combines:
- **Behavioral Questionnaire**
- **Dining Habit Analysis**
- **Speech Analysis**
- **Eye-Tracking Gaze Test**

to estimate autism risk.  
⚠️ This is **not** a clinical diagnosis.
""")

# =======================
# UTILITIES & CACHING
# =======================
@st.cache_resource
def load_behavior_model(path="behavior_model.pkl"):
    return joblib.load(path)

@st.cache_resource
def load_speech_model(path="speech_model.h5"):
    # compile=False for maximum compatibility across TF/Keras minor versions
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
    model = tf.keras.models.load_model(path, compile=False)
    return model

def extract_mfcc_spectrogram(file_path, sr_target=16000, n_mfcc=40, max_frames=216):
    """
    Load audio, compute MFCCs (40 x T), then pad/truncate to (40 x max_frames),
    and reshape to (1, 40, 216, 1) for a typical CNN(+LSTM) front-end.

    This fixes the 'Negative dimension' Conv2D error by ensuring a valid 2D input.
    """
    # Load and resample
    y, sr = librosa.load(file_path, sr=sr_target, mono=True)

    # Safety: if silent or too short, pad a little to avoid empty features
    if y is None or len(y) == 0:
        y = np.zeros(int(0.5 * sr_target), dtype=np.float32)

    # MFCC shape: (n_mfcc, T)
    # Use standard params; hop_length ~10ms gives reasonable temporal resolution
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160, win_length=400)
    # Normalize per-coefficient (optional but often helps)
    mfcc = librosa.util.normalize(mfcc)

    # Pad/Truncate along time axis to fixed max_frames
    T = mfcc.shape[1]
    if T < max_frames:
        pad = max_frames - T
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]

    # Reshape -> (1, 40, 216, 1)
    x = mfcc[..., np.newaxis]       # (40, 216, 1)
    x = np.expand_dims(x, axis=0)   # (1, 40, 216, 1)
    return x

def predict_speech_probability(model, file_path):
    """Return probability (float) from speech model; assumes 1 output neuron (sigmoid)."""
    x = extract_mfcc_spectrogram(file_path)
    pred = model.predict(x, verbose=0)
    # Handle model output shape robustly
    prob = float(np.ravel(pred)[0])
    # Clamp just in case
    prob = float(np.clip(prob, 0.0, 1.0))
    return prob

def run_eye_tracking():
    # Launch your OpenCV app with the same Python executable as Streamlit
    subprocess.run([sys.executable, "app.py"])

def load_gaze_accuracy(csv_path):
    """Return gaze accuracy (mean of `is_target`) from a log file, or None if missing/invalid."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or "is_target" not in df.columns:
        return None
    usable = df[~df["blink"]] if "blink" in df.columns else df
    if usable.empty:
        return 0.0
    return float(usable["is_target"].mean())

# =======================
# LOAD MODELS
# =======================
try:
    behavior_model = load_behavior_model("behavior_model.pkl")
except Exception as e:
    st.error(f"Failed to load behavior model: {e}")
    st.stop()

try:
    speech_model = load_speech_model("speech_model.h5")
except Exception as e:
    st.error(f"Failed to load speech model (speech_model.h5): {e}")
    st.info("Ensure the file exists and matches your TensorFlow/Keras version.")
    st.stop()

# =======================
# STEP 1: BEHAVIORAL QUESTIONS
# =======================
st.header("Step 1: Behavioral Questionnaire")

behavior_questions = [
    "1. Does the child avoid eye contact?",
    "2. Does the child prefer playing alone?",
    "3. Does the child repeat certain actions or words?",
    "4. Does the child have difficulty understanding feelings?",
    "5. Does the child resist changes in routine?",
    "6. Does the child have unusual attachments to objects?",
    "7. Does the child show extreme distress over small changes?",
    "8. Does the child have trouble making friends?",
    "9. Does the child focus on parts of objects?",
    "10. Does the child have delayed speech or language skills?"
]

behavioral_responses = []
for q in behavior_questions:
    # 1 = Yes (indicative), 0 = No
    ans = st.radio(q, [0, 1], index=0, horizontal=True)
    behavioral_responses.append(ans)

behavior_score = float(behavior_model.predict_proba([behavioral_responses])[0][1])

# =======================
# STEP 2: DINING HABITS
# =======================
st.header("Step 2: Dining Habits Questionnaire")
st.caption("Rate each from 0 (Never) to 4 (Always)")

dining_questions = [
    "Prefers food of specific color, texture, or temperature",
    "Resists trying new foods",
    "Distressed by sound of chewing or utensils",
    "Insists on same plate or cup",
    "Lines up food or arranges it in patterns",
    "Requires strict mealtime routines",
    "Eats fewer than 10 different foods",
    "Avoids touching certain textures",
    "Meltdowns related to meals",
    "Gags or vomits with certain textures"
]

dining_responses = []
for dq in dining_questions:
    score = st.slider(dq, 0, 4, 0)
    dining_responses.append(score)

max_dining_score = len(dining_responses) * 4
dining_score = float(sum(dining_responses) / max_dining_score) if max_dining_score > 0 else 0.0

# =======================
# STEP 3: SPEECH ANALYSIS
# =======================
st.header("Step 3: Speech Analysis")
st.write("Upload a speech sample **or** record directly.")

SR = 16000  # sample rate for recording / processing

col_u, col_r = st.columns(2)
with col_u:
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg", "m4a"])
with col_r:
    # User controls duration — no fixed limit enforced (but keep reasonable)
    rec_secs = st.number_input("Record duration (seconds)", min_value=1, max_value=120, value=5, step=1)
    record_now = st.button("🎙️ Record Audio")

audio_path = None

if uploaded_audio is not None:
    # Save to temp WAV for librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
    st.success("Audio uploaded.")
elif record_now:
    st.info(f"Recording for {rec_secs} seconds...")
    try:
        recording = sd.rec(int(rec_secs * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, recording, SR)
            audio_path = tmp.name
        st.success("Recording complete.")
    except Exception as e:
        st.error(f"Recording failed: {e}")

speech_score = None
if audio_path:
    try:
        speech_score = predict_speech_probability(speech_model, audio_path)
        st.success(f"**Speech-based Autism Probability:** {speech_score:.2f}")
    except Exception as e:
        st.error(f"Speech analysis failed: {e}")

# =======================
# STEP 4: EYE TRACKING TEST
# =======================
st.header("Step 4: Eye-Tracking Gaze Test")
st.write("Click below to run the eye-tracking test. This will open your webcam and play both stimuli.")

if st.button("▶ Run Eye-Tracking Test"):
    run_eye_tracking()

# Per-stimulus logs produced by your app.py
stimuli_logs = [
    ("bio_vs_scrambled", "output/gaze_log_bio_vs_scrambled.csv"),
    ("shapes_vs_social", "output/gaze_log_shapes_vs_social.csv"),
]

gaze_scores = []
for name, path in stimuli_logs:
    score = load_gaze_accuracy(path)
    if score is not None:
        st.write(f"**Gaze Accuracy ({name}):** {score:.2f}")
        gaze_scores.append(score)
    else:
        st.warning(f"No gaze data for {name}.")

gaze_score = float(sum(gaze_scores) / len(gaze_scores)) if gaze_scores else None
if gaze_score is not None:
    st.success(f"**Combined Gaze Score:** {gaze_score:.2f}")
else:
    st.info("Run the eye-tracking test to calculate gaze score.")

# =======================
# STEP 5: FINAL COMBINED SCORE
# =======================
st.header("Final Results")

# Show intermediate scores even if one modality is missing
st.write(f"**Behavioral Score:** {behavior_score:.2f}")
st.write(f"**Dining Habits Score:** {dining_score:.2f}")
if speech_score is not None:
    st.write(f"**Speech Score:** {speech_score:.2f}")
else:
    st.write("**Speech Score:** _not available_")
if gaze_score is not None:
    st.write(f"**Gaze Score:** {gaze_score:.2f}")
else:
    st.write("**Gaze Score:** _not available_")

# Only compute final if all three modal scores are present
if (speech_score is not None) and (gaze_score is not None):
    # You can adjust weights later if needed
    final_risk_score = (
        0.40 * behavior_score +
        0.20 * dining_score +
        0.15 * speech_score +
        0.25 * gaze_score
    )
    st.write(f"**Final Autism Risk Score:** {final_risk_score:.2f}")

    if final_risk_score > 0.60:
        st.error("🔴 High Risk - Further clinical evaluation recommended.")
    elif final_risk_score > 0.40:
        st.warning("🟠 Moderate Risk - Monitor and consider professional screening.")
    else:
        st.success("🟢 Low Risk - No immediate concern.")
else:
    st.info("Provide speech and gaze signals to compute the **Final Autism Risk Score**.")
