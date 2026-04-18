"""
models.py
─────────
Two model architectures:
  1. CNN Trend Scout  – 1-D CNN (TensorFlow/Keras) for Bullish/Bearish prediction
  2. SVM Risk Officer – RBF-kernel SVM for Safe/Risky classification

Handles class imbalance via class_weight (CNN) and SMOTE (SVM).
"""

from __future__ import annotations
import numpy as np
import joblib
import os

# ── TensorFlow (lazy import so SVM works even without GPU) ────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Imbalanced-learn (SMOTE) ──────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    print("[models] imbalanced-learn not found – falling back to class_weight for SVM")


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

CNN_PATH = os.path.join(MODELS_DIR, "cnn_trend.keras")
SVM_PATH = os.path.join(MODELS_DIR, "svm_risk.pkl")


# ─────────────────────────────────────────────────────────────────────────────
#  1-D CNN  – Trend Scout
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(input_shape: tuple[int, int]) -> keras.Model:
    """
    input_shape = (window, n_features)  e.g. (30, 7)
    Output: sigmoid probability of Bullish (class 1).
    """
    inp = keras.Input(shape=input_shape, name="ohlcv_window")

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="trend")(x)

    model = keras.Model(inp, out, name="CNN_TrendScout")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
) -> keras.Model:
    print("[CNN] Training …")
    # class weights for imbalance
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), cw.tolist()))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )

    model = build_cnn(input_shape=(X_train.shape[1], X_train.shape[2]))
    cb = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
    ]
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1,
    )
    model.save(CNN_PATH)
    print(f"[CNN] Saved → {CNN_PATH}")
    return model


def load_cnn() -> keras.Model:
    return keras.models.load_model(CNN_PATH)


def predict_cnn(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Return +1 (Bullish) or -1 (Bearish) per sample."""
    probs = model.predict(X, verbose=0).flatten()
    return np.where(probs >= 0.5, 1, -1)


def predict_cnn_proba(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0).flatten()


# ─────────────────────────────────────────────────────────────────────────────
#  SVM  – Risk Officer
# ─────────────────────────────────────────────────────────────────────────────

def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    print("[SVM] Training …")
    X_t, y_t = X_train, y_train

    if _SMOTE_AVAILABLE:
        try:
            sm = SMOTE(random_state=42)
            X_t, y_t = sm.fit_resample(X_train, y_train)
            print(f"[SVM] SMOTE resampled: {dict(zip(*np.unique(y_t, return_counts=True)))}")
        except Exception as e:
            print(f"[SVM] SMOTE skipped ({e}); using class_weight")
            X_t, y_t = X_train, y_train

    clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    clf.fit(X_t, y_t)
    joblib.dump(clf, SVM_PATH)
    print(f"[SVM] Saved → {SVM_PATH}")
    return clf


def load_svm() -> SVC:
    return joblib.load(SVM_PATH)


def predict_svm(clf: SVC, X: np.ndarray) -> np.ndarray:
    """Return 1 (Safe) or 0 (Risky) per sample."""
    return clf.predict(X)


def predict_svm_proba(clf: SVC, X: np.ndarray) -> np.ndarray:
    """Return P(Safe) in [0, 1]."""
    return clf.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cnn(model: keras.Model, X_test, y_test):
    preds = predict_cnn(model, X_test)
    y_binary = ((preds + 1) // 2).astype(int)
    print("\n[CNN] Classification Report:")
    print(classification_report(y_test, y_binary, target_names=["Bearish", "Bullish"]))


def evaluate_svm(clf: SVC, X_test, y_test):
    preds = predict_svm(clf, X_test)
    print("\n[SVM] Classification Report:")
    print(classification_report(y_test, preds, target_names=["Risky", "Safe"]))