import os
import glob
import numpy as np
import pandas as pd
import librosa

# -----------------------------------------------------------
# PATHS (CHANGE IF NEEDED)
# -----------------------------------------------------------
DATA_ROOT = r"Data"      # Your dataset root folder
SPLITS = ["Train", "Eval", "Test"]  # Your folders

# -----------------------------------------------------------
# AUDIO SETTINGS
# -----------------------------------------------------------
SAMPLE_RATE = 22050
N_MFCC = 20
N_CHROMA = 12             # For chromagram

# -----------------------------------------------------------
# ADVANCED FEATURE EXTRACTOR
# -----------------------------------------------------------
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # -------------------- MFCC --------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # ---------------- Spectral features ----------------
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    # ---------------- Chromagram ----------------
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)

    # ---------------- STATISTICS ----------------
    def stats(x):
        return np.mean(x), np.std(x)

    features = []

    # MFCC stats
    for i in range(N_MFCC):
        m_mean, m_std = stats(mfcc[i])
        features += [m_mean, m_std]

    # Delta MFCC stats
    for i in range(N_MFCC):
        d_mean, d_std = stats(mfcc_delta[i])
        features += [d_mean, d_std]

    # Delta-Delta MFCC stats
    for i in range(N_MFCC):
        dd_mean, dd_std = stats(mfcc_delta2[i])
        features += [dd_mean, dd_std]

    # Spectral features stats
    for arr in [centroid, bandwidth, rolloff, zcr]:
        m, s = stats(arr[0])
        features += [m, s]

    # Chromagram features stats
    for i in range(N_CHROMA):
        c_mean, c_std = stats(chroma[i])
        features += [c_mean, c_std]

    return np.array(features)


# -----------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------
rows = []

for split in SPLITS:
    split_dir = os.path.join(DATA_ROOT, split)
    genres = sorted(os.listdir(split_dir))

    for genre in genres:
        genre_dir = os.path.join(split_dir, genre)

        audio_files = glob.glob(os.path.join(genre_dir, "*.wav"))

        for file_path in audio_files:
            try:
                feat = extract_features(file_path)
            except Exception as e:
                print(f"ERROR in {file_path}: {e}")
                continue

            row = {"split": split,
                   "genre": genre,
                   "file": os.path.basename(file_path)}

            # Add feature columns
            for i, value in enumerate(feat):
                row[f"f{i+1}"] = value

            rows.append(row)

# -----------------------------------------------------------
# SAVE DATA
# -----------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv("PRO_Features.csv", index=False)
df.to_excel("PRO_Features.xlsx", index=False)

print("\nDONE! Saved:")
print(" - PRO_Features.csv")
print(" - PRO_Features.xlsx\n")
