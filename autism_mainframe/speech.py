import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# ====================
# SETTINGS
# ====================
DATASET_DIR = "data"  # folder with subfolders ASD/ and TD/
SAMPLE_RATE = 22050
DURATION = 10  # seconds
N_MFCC = 40

# ====================
# FEATURE EXTRACTION
# ====================
def extract_features(file_path, max_pad_len=216):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        # pad or truncate to fixed length
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error extracting features:", e)
        return None

# ====================
# LOAD DATASET
# ====================
labels = {"ASD":0, "TD":1}  # ASD vs Typical Development
X, y = [], []

for label, idx in labels.items():
    folder = os.path.join(DATASET_DIR, label)
    for file in tqdm(os.listdir(folder), desc=f"Loading {label}"):
        path = os.path.join(folder, file)
        features = extract_features(path)
        if features is not None:
            X.append(features)
            y.append(idx)

X = np.array(X)
y = to_categorical(np.array(y), num_classes=2)

# reshape for CNN (samples, height, width, channels)
X = X[..., np.newaxis]

print("Dataset shape:", X.shape, y.shape)

# ====================
# TRAIN-TEST SPLIT
# ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

# ====================
# MODEL
# ====================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(N_MFCC, X.shape[2], 1)),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ====================
# TRAIN
# ====================
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=30, batch_size=32)

# save model
model.save("speech_model.h5")
print("✅ Speech model saved as speech_model.h5")
