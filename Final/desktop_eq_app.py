import sys
import os
import numpy as np
import librosa
import joblib

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
SAMPLE_RATE = 22050
N_MFCC = 20
N_CHROMA = 12

# ---------------------------------------------------------
# LOAD MODEL + LABEL ENCODER
# ---------------------------------------------------------
try:
    model = joblib.load("svm_genre_model_PRO.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    print("Error loading model or label encoder:", e)
    sys.exit(1)

# ---------------------------------------------------------
# GENRE EQ PRESETS  (32, 64, 125, 250, 500, 1k, 2k, 4k, 8k, 16k)
# ---------------------------------------------------------
EQ_PRESETS = {
    "rock":      [+60, +10, -20,   0,   0, -20,   0, -20,   0, +60],
    "jazz":      [  0,   0,   0, +10, +10, -50,  -5,   0,   0,   0],
    "pop":       [-20,   5, -80, -10,   0,   0,   0, -50,   5,   0],
    "classical": [  0,   0,   0,   0,   0, +50, +10, +100, +20, +100],
    "hiphop":    [+50,   0, -50,   0,   0, -50,   0,   0,   0, +20],
    "blues":     [+20, +20,-100, -20, +50, +10,   0, +10,   0, +20],
    "metal":     [+25,   0,   0, -10,  -5, -60,   0,   0,   0,   0],
}

BANDS = [
    "32 Hz", "64 Hz", "125 Hz", "250 Hz", "500 Hz",
    "1 kHz", "2 kHz", "4 kHz", "8 kHz", "16 kHz"
]

# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
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
        m, s = stats(mfcc[i]);        features += [m, s]
    for i in range(N_MFCC):
        m, s = stats(mfcc_delta[i]);  features += [m, s]
    for i in range(N_MFCC):
        m, s = stats(mfcc_delta2[i]); features += [m, s]

    for arr in [centroid, bandwidth, rolloff, zcr]:
        m, s = stats(arr[0]);         features += [m, s]

    for i in range(N_CHROMA):
        m, s = stats(chroma[i]);      features += [m, s]

    return np.array(features).reshape(1, -1)

# ---------------------------------------------------------
# CUSTOM SLIDER (grey by default, green segment from 0 to value)
# ---------------------------------------------------------
class EQSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        # neutral style: grey groove & handle, no green from stylesheet
        self.setStyleSheet("""
            QSlider::groove:vertical {
                background: #333333;
                width: 4px;
                border-radius: 2px;
            }
            QSlider::handle:vertical {
                background: #aaaaaa;
                border: 1px solid #aaaaaa;
                height: 18px;
                margin: -4px -8px;
                border-radius: 9px;
            }
            QSlider::sub-page:vertical {
                background: #333333;
                border-radius: 2px;
            }
            QSlider::add-page:vertical {
                background: #333333;
                border-radius: 2px;
            }
        """)

    def paintEvent(self, event):
        # First let Qt draw the normal slider (all grey)
        super().paintEvent(event)

        v = self.value()
        if v == 0:
            # at 0 we want no green at all
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        margin_top = 16
        margin_bottom = 16
        x = rect.center().x()

        min_val, max_val = self.minimum(), self.maximum()
        height = rect.height() - margin_top - margin_bottom

        def value_to_y(val):
            # map val in [min,max] -> y coordinate
            ratio = (val - min_val) / float(max_val - min_val)
            return int(rect.bottom() - margin_bottom - ratio * height)

        zero_y = value_to_y(0)
        curr_y = value_to_y(v)

        # choose segment from 0 to current value
        if v > 0:
            y1, y2 = curr_y, zero_y
        else:
            y1, y2 = zero_y, curr_y

        pen = QPen(QColor("#1db954"), 4, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(x, y1, x, y2)
        painter.end()

# ---------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------
class GenreEQApp(QWidget):
    def __init__(self):
        super().__init__()

        self.sliders = []
        self.value_labels = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AI Music Genre Detection - Desktop")
        self.setMinimumSize(900, 550)

        # ---------- DARK THEME BASE ----------
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #f5f5f5;
                font-family: Segoe UI, Arial;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#SubtitleLabel {
                font-size: 16px;
                color: #d0d0d0;
            }
            QLabel#GenreLabel {
                font-size: 20px;
                font-weight: 600;
                margin-top: 10px;
            }
            QLabel#FileLabel {
                font-size: 15px;
                color: #bbbbbb;
            }
            QPushButton {
                background-color: #1db954;
                color: #ffffff;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: 500;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1ed760;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 10px;
                font-size: 16px;
                font-weight: 500;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(10)

        # ---------- Header ----------
        title_label = QLabel("🎵 AI Music Genre Detection")
        title_label.setObjectName("TitleLabel")
        main_layout.addWidget(title_label)

        subtitle_label = QLabel("Upload a .wav file to detect its genre and see the adaptive equalizer preset.")
        subtitle_label.setObjectName("SubtitleLabel")
        main_layout.addWidget(subtitle_label)

        # ---------- File row ----------
        file_row = QHBoxLayout()
        file_row.setSpacing(10)

        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("FileLabel")

        browse_btn = QPushButton("Select WAV File")
        browse_btn.clicked.connect(self.choose_file)

        file_row.addWidget(self.file_label)
        file_row.addStretch()
        file_row.addWidget(browse_btn)

        main_layout.addLayout(file_row)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #333333;")
        main_layout.addWidget(line)

        # ---------- Genre label ----------
        self.genre_label = QLabel("Predicted Genre: ---")
        self.genre_label.setObjectName("GenreLabel")
        main_layout.addWidget(self.genre_label)

        # ---------- Equalizer group ----------
        eq_group = QGroupBox("Equalizer (UI Demo Only)")
        eq_group.setFixedHeight(320)
        eq_layout = QHBoxLayout()
        eq_layout.setSpacing(24)

        for band_name in BANDS:
            band_layout = QVBoxLayout()
            band_layout.setSpacing(6)

            # numeric value above slider
            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignHCenter)
            value_label.setStyleSheet("font-size: 14px; color: #f0f0f0;")
            self.value_labels.append(value_label)

            slider = EQSlider()
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setSingleStep(5)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setTickInterval(50)
            slider.setFixedHeight(180)
            slider.valueChanged.connect(self.update_value_labels)
            self.sliders.append(slider)

            band_label = QLabel(band_name)
            band_label.setAlignment(Qt.AlignHCenter)
            band_label.setStyleSheet("font-size: 14px; color: #e0e0e0;")

            band_layout.addWidget(value_label)
            band_layout.addWidget(slider, stretch=1, alignment=Qt.AlignHCenter)
            band_layout.addWidget(band_label)

            eq_layout.addLayout(band_layout)

        eq_group.setLayout(eq_layout)
        main_layout.addWidget(eq_group)

        # Info label
        info_label = QLabel(
            "Note: Sliders are for visual demonstration only and do not modify the audio.\n"
            "When a genre is detected, sliders jump to that genre's preset, but you can still adjust them manually."
        )
        info_label.setStyleSheet("font-size: 13px; color: #b0b0b0;")
        main_layout.addWidget(info_label)

        main_layout.addStretch()
        self.setLayout(main_layout)

        self.update_value_labels()

    # ---------- slider labels ----------
    def update_value_labels(self):
        for label, slider in zip(self.value_labels, self.sliders):
            label.setText(str(slider.value()))

    # ---------- File handling ----------
    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WAV File",
            "",
            "Audio Files (*.wav)"
        )
        if not file_path:
            return

        self.file_label.setText(os.path.basename(file_path))

        try:
            features = extract_features(file_path)
            pred_class = model.predict(features)[0]
            genre_name = label_encoder.inverse_transform([pred_class])[0]
        except Exception as e:
            self.genre_label.setText("Error processing file")
            print("Error during prediction:", e)
            return

        self.genre_label.setText(f"Predicted Genre: {genre_name.upper()}")

        genre_key = genre_name.lower()
        if genre_key in EQ_PRESETS:
            preset = EQ_PRESETS[genre_key]
            for i, slider in enumerate(self.sliders):
                if i < len(preset):
                    slider.setValue(preset[i])

        self.update_value_labels()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenreEQApp()
    window.show()
    sys.exit(app.exec_())
