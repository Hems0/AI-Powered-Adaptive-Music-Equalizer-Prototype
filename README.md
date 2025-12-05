🎵 AI Music Genre Detection + Adaptive Equalizer (Prototype)

A machine learning–powered audio project that predicts music genres and automatically adjusts an equalizer UI.


🌟 Overview

This project explores how "AI can enhance the listening experience" by automatically detecting the genre of a song and adjusting a "10-band equalizer preset" to match the style of music.

The system uses:

* "Machine Learning" (SVM) to classify genres
* "Audio signal processing" (MFCC, spectral features, etc.)
* A "Streamlit web app" for a clean UI
* A "PyQt5 desktop application" for a real software experience
* A visual-only "adaptive equalizer" that changes based on the predicted genre

The idea comes from a simple pain point:

> Different genres sound best with different EQ presets — but switching manually breaks the experience.

This prototype demonstrates how AI can automate that.

🎯 Key Features

🧠 AI Genre Classification

The system predicts one of 7 genres:

* Blues
* Classical
* Hip-Hop
* Jazz
* Metal
* Pop
* Rock

Using a tuned SVM model trained on the "GTZAN dataset", the final accuracy reached:

⭐ **~77% Test Accuracy (with PRO features)

🎚 Adaptive Equalizer (UI Prototype Only)

* Full **10-band equalizer** (32 Hz → 16 kHz)
* Automatically adjusts sliders to match the detected genre
* User can still manually modify sliders
* This equalizer is for **UI demonstration** only — it does *not* modify audio yet

Genre presets include Rock, Pop, Jazz, Classical, Hip-Hop, Metal, and Blues.



💻 Two Application Versions Included

1️⃣ Web App — Streamlit

* Upload `.wav` songs
* Plays audio
* Predicts genre
* Updates equalizer sliders
* Clean, dark-themed UI

Run using:

```
streamlit run web_eq.py
```

2️⃣ Desktop App — PyQt5

* Equalizer UI in classic desktop style
* Real-application feel
* Useful for future integration with system audio

Run using:

```
python desktop_eq_app.py
```

🔬How the AI Works

Feature Extraction

The model uses a rich set of audio features:

✔ 20 MFCC coefficients
✔ 20 Delta MFCC
✔ 20 Delta-Delta MFCC
✔ Spectral Centroid
✔ Spectral Bandwidth
✔ Spectral Rolloff
✔ Zero Crossing Rate
✔ 12-bin Chroma Vector

These features were extracted using `librosa`.

Model Training

The training scripts are inside the "Training/" folder:

* `mfcc.py` – MFCC-only feature extraction
* `SMV.py` – SVM model training + tuning
* `MFCC_Features.xlsx`
* `PRO_Features.xlsx`
* `svm_genre_model.pkl`
* `svm_genre_model_tuned.pkl`

The final model used in the app is:

```
svm_genre_model_PRO.pkl
```

🎵 Testing the App
Sample test audio files are included inside the `Test_Audio` folder.
These are 30-second clips from the GTZAN dataset (educational use only).
You can upload any of them to see how the AI predicts the genre and adjusts the equalizer UI.


🔮 Future Improvements

This prototype opens the door for more advanced versions:

🎧 Real-Time System Audio Listening

A future version could allow the AI to:

* Listen to what the user is playing (Spotify, YouTube, Netflix, games, etc.)
* Predict genre/content type on the fly
* Automatically switch EQ presets
* Turn itself OFF automatically during Zoom/Meet/calls for privacy
* Provide a toggle for full user control

Capturing system audio is low-level and OS-dependent, so this feature is planned for future development.

---

📁Project Structure

```
/
│── FINAL/
│     ├── desktop_eq_app.py
│     ├── web_eq.py
│     ├── svm_genre_model_PRO.pkl
│     ├── label_encoder.pkl
│
│── Training/
│     ├── mfcc.py
│     ├── SMV.py
│     ├── MFCC_Features.xlsx
│     ├── PRO_Features.xlsx
│     ├── svm_genre_model.pkl
│     ├── svm_genre_model_tuned.pkl
│
│── requirements.txt
│── README.md
```

🛠Installation

Install dependencies:

```
pip install -r requirements.txt
```

🚀Usage

▶ Run the Web App
```
streamlit run web_eq.py
```

▶ Run the Desktop App
```
python desktop_eq_app.py
```

 ❤️Why I Built This

As a music lover using IEMs and headphones, I found that manually switching EQ presets for every song or genre was tedious.
This project began as a personal idea:

> “What if AI could detect what I’m listening to and adjust the EQ automatically?”

This prototype explores that vision — combining "audio signal processing", "machine learning", and "modern UI design".

📌License
This project is for research and demonstration purposes.


