import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# -----------------------------
# 0. CONFIG
# -----------------------------
CSV_FILE = "PRO_Features.csv"
MODEL_OUT = "svm_genre_model_PRO.pkl"
LABEL_ENCODER_OUT = "label_encoder.pkl"

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_FILE)
print("\nLoaded:", CSV_FILE)
print("Total rows:", len(df))

# -----------------------------
# 2. QUICK DIAGNOSTICS: counts per split and per genre
# -----------------------------
print("\nCounts per split:")
print(df["split"].value_counts())

print("\nCounts per genre (overall):")
print(df["genre"].value_counts())

print("\nCounts per genre per split:")
print(df.groupby(["split", "genre"]).size().unstack(fill_value=0))

# -----------------------------
# 3. SELECT NUMERIC FEATURE COLUMNS ONLY
# -----------------------------
non_feature_cols = ["split", "genre", "file"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found. Make sure PRO_Features.csv contains feature columns named like f1, f2, ...")

print("\nTotal numeric features found:", len(feature_cols))
print("Example feature columns:", feature_cols[:10])

# -----------------------------
# 4. SPLIT DATA (Train+Eval vs Test)
# -----------------------------
train_eval_df = df[df["split"].isin(["Train", "Eval"])]
test_df       = df[df["split"] == "Test"]

X_train_eval = train_eval_df[feature_cols].values
y_train_eval = train_eval_df["genre"].values

X_test = test_df[feature_cols].values
y_test = test_df["genre"].values

# -----------------------------
# 5. LABEL ENCODING
# -----------------------------
label_encoder = LabelEncoder()
y_train_eval_enc = label_encoder.fit_transform(y_train_eval)

# Handle potential missing classes in test: transform but do not assume all classes present
# We'll transform y_test and handle cases where some classes are absent in the test split.
y_test_enc = label_encoder.transform(y_test)

classes = list(label_encoder.classes_)
print("\nClasses:", classes)

# -----------------------------
# 6. BUILD PIPELINE (Scaler + SVM)
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])

# -----------------------------
# 7. HYPERPARAM GRID
# -----------------------------
param_grid = {
    "svc__kernel": ["rbf"],
    "svc__C": [1, 10, 50, 100],
    "svc__gamma": ["scale", 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# -----------------------------
# 8. TRAIN + TUNE
# -----------------------------
print("\nTraining + tuning SVM (this may take a few minutes)...")
grid.fit(X_train_eval, y_train_eval_enc)
print("Done.")

print("\nBest parameters:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)

best_model = grid.best_estimator_

# -----------------------------
# 9. EVALUATE ON TEST SET
# -----------------------------
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test_enc, y_pred)

print("\n=== FINAL TEST RESULTS ===")
print("Test accuracy:", test_accuracy)

# Ensure classification_report always lists all classes in the encoder (even if missing in test)
all_labels = list(range(len(classes)))
all_names = classes

print("\nClassification Report (all classes shown):")
print(classification_report(y_test_enc, y_pred, labels=all_labels, target_names=all_names, zero_division=0))

print("\nConfusion Matrix (rows=true, cols=pred) with all classes:")
cm = confusion_matrix(y_test_enc, y_pred, labels=all_labels)
print(cm)

# -----------------------------
# 10. SAVE MODEL + ENCODER
# -----------------------------
joblib.dump(best_model, MODEL_OUT)
joblib.dump(label_encoder, LABEL_ENCODER_OUT)

print(f"\nSaved model -> {MODEL_OUT}")
print(f"Saved label encoder -> {LABEL_ENCODER_OUT}")
print("\nAll done.")
