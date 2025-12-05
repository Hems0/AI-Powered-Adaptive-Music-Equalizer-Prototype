import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ---------------------------------------------------------
# 1. LOAD PRO FEATURE DATA
# ---------------------------------------------------------
df = pd.read_csv("PRO_Features.csv")

# ---------------------------------------------------------
# 2. SELECT FEATURE COLUMNS (NUMERIC ONLY)
# ---------------------------------------------------------
non_feature_cols = ["split", "genre", "file"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

print("\nTotal numeric features found:", len(feature_cols))
print("Example feature columns:", feature_cols[:10])

# ---------------------------------------------------------
# 3. SPLIT DATA (TRAIN+EVAL vs TEST)
# ---------------------------------------------------------
train_eval_df = df[df["split"].isin(["Train", "Eval"])]
test_df       = df[df["split"] == "Test"]

X_train_eval = train_eval_df[feature_cols].values
y_train_eval = train_eval_df["genre"].values

X_test = test_df[feature_cols].values
y_test = test_df["genre"].values

# ---------------------------------------------------------
# 4. LABEL ENCODER
# ---------------------------------------------------------
label_encoder = LabelEncoder()
y_train_eval_enc = label_encoder.fit_transform(y_train_eval)
y_test_enc       = label_encoder.transform(y_test)

print("\nClasses:", list(label_encoder.classes_))

# ---------------------------------------------------------
# 5. PIPELINE (SCALER + SVM)
# ---------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])

# ---------------------------------------------------------
# 6. HYPERPARAMETER GRID
# ---------------------------------------------------------
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
    n_jobs=-1
)

print("\nTraining + tuning SVM... (This may take a few minutes)")
grid.fit(X_train_eval, y_train_eval_enc)
print("Done.")

print("\nBest parameters:", grid.best_params_)
print("Best training CV accuracy:", grid.best_score_)

best_model = grid.best_estimator_

# ---------------------------------------------------------
# 7. FINAL TEST EVALUATION
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test_enc, y_pred)
print("\n=== FINAL TEST RESULTS (PRO FEATURES + TUNED SVM) ===")
print("Accuracy:", test_accuracy)
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

# ---------------------------------------------------------
# 8. SAVE MODEL + LABEL ENCODER
# ---------------------------------------------------------
joblib.dump(best_model, "svm_genre_model_PRO.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nSaved model as: svm_genre_model_PRO.pkl")
print("Saved label encoder as: label_encoder.pkl")
print("\nTraining complete!\n")
