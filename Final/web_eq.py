import streamlit as st
import librosa
import numpy as np
import joblib
import os

SAMPLE_RATE = 22050
N_MFCC = 20
N_CHROMA = 12


BASE_DIR = Path(__file__).resolve().parent


model = joblib.load("(BASE_DIR /svm_genre_model_PRO.pkl")
label_encoder = joblib.load("(BASE_DIR /label_encoder.pkl")


EQ_PRESETS = {
    "rock":      [+60, +10, -20, 0, 0, -20, 0, -20, 0, +60],
    "jazz":      [0, 0, 0,  +10,  +10, -50, -5,  0, 0, 0],
    "pop":       [-20, 5, -80, -10,  0, 0, 0, -50, 5, 0],
    "classical": [0, 0,   0, 0, 0, +50 +10,  +100, +20, +100],
    "hiphop":    [+50, 0, -50, 0, 0, -50,  0, 0, 0, +20],
    "blues":     [+20, +20, -100,  -20, +50, +10, 0, +10, 0,   +20],
    "metal":     [+25, 0, 0, -10, -5, -60, 0, 0, 0, 0],
}

def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

 
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)


    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)

    def stats(arr):
        return np.mean(arr), np.std(arr)

    features = []

   
    for i in range(N_MFCC):
        m, s = stats(mfcc[i])
        features += [m, s]

  
    for i in range(N_MFCC):
        m, s = stats(mfcc_delta[i])
        features += [m, s]

    for i in range(N_MFCC):
        m, s = stats(mfcc_delta2[i])
        features += [m, s]


    for arr in [centroid, bandwidth, rolloff, zcr]:
        m, s = stats(arr[0])
        features += [m, s]

  
    for i in range(N_CHROMA):
        m, s = stats(chroma[i])
        features += [m, s]

    return np.array(features).reshape(1, -1)


st.set_page_config(page_title="AI Music Genre Detection", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
        }
        .eq-label {
            text-align: center;
            font-size: 0.8rem;
            color: #cccccc;
        }
    </style>
""", unsafe_allow_html=True)

bands = [
    "32 Hz", "64 Hz", "125 Hz", "250 Hz", "500 Hz",
    "1 kHz", "2 kHz", "4 kHz", "8 kHz", "16 kHz"
]

if "last_genre" not in st.session_state:
    st.session_state["last_genre"] = None

for i in range(len(bands)):
    key = f"eq_{i}"
    if key not in st.session_state:
        st.session_state[key] = 0


st.title("🎵 AI Music Genre Detection")
st.write("Upload a **.wav** song and let the AI predict its genre!")

uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

predicted_genre = None

if uploaded_file is not None:

    temp_path = "temp_uploaded.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

 
    st.audio(temp_path)

    with st.spinner("Extracting features..."):
        features = extract_features(temp_path)

    with st.spinner("Predicting genre..."):
        pred_class = model.predict(features)[0]
        predicted_genre = label_encoder.inverse_transform([pred_class])[0]

    st.success(f"🎧 **Predicted Genre: {predicted_genre.upper()}**")

    genre_key = predicted_genre.lower()
    if st.session_state["last_genre"] != predicted_genre:
        st.session_state["last_genre"] = predicted_genre
        if genre_key in EQ_PRESETS:
            preset = EQ_PRESETS[genre_key]
            for i, val in enumerate(preset):
                st.session_state[f"eq_{i}"] = val

    
    os.remove(temp_path)


st.markdown("---")
st.subheader("🎚 Equalizer (UI Demo Only)")
st.caption(
    "When the AI predicts a genre, the sliders jump to that genre's EQ preset. "
    "You can still move them manually. This prototype only changes the **visual sliders** "
    "and does **not** modify the sound."
)

cols = st.columns(len(bands))
eq_values = {}

for i, (col, band) in enumerate(zip(cols, bands)):
    with col:
        st.markdown(f"<div class='eq-label'>{band}</div>", unsafe_allow_html=True)
        eq_values[band] = st.slider(
            label=" ",
            min_value=-100,
            max_value=100,
            value=st.session_state[f"eq_{i}"],  
            step=5,
            key=f"eq_{i}",
        )

# with st.expander("Show equalizer values (for report/demo)"):
#     st.write(eq_values)


