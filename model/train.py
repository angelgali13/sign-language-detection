#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  Sign Language Detection — Training Script
#
#  Usage:
#      python model/train.py
#
#  Expects model/X.npy, model/y.npy to exist
#  (run data_collection/preprocess.py first).
# ─────────────────────────────────────────────

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              multilabel_confusion_matrix)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (MODEL_DIR, SIGNS, EPOCHS, BATCH_SIZE,
                    VALIDATION_SPLIT, MODEL_PATH)
from model.model import build_model, get_callbacks


# ── Load data ──────────────────────────────────
def load_data():
    X_path = os.path.join(MODEL_DIR, "X.npy")
    y_path = os.path.join(MODEL_DIR, "y.npy")
    if not os.path.exists(X_path):
        raise FileNotFoundError(
            "X.npy not found. Run  data_collection/preprocess.py  first."
        )
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"[✓] Loaded  X{X.shape}  y{y.shape}")
    return X, y


# ── Plot helpers ───────────────────────────────
def plot_training_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Saved training plot  →  {path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Saved confusion matrix  →  {path}")
    plt.close()


# ── Main ───────────────────────────────────────
def train():
    X, y = load_data()
    signs = list(np.load(os.path.join(MODEL_DIR, "labels.npy"),
                         allow_pickle=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=np.argmax(y, axis=1),
    )
    print(f"    Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

    # Build
    model = build_model(num_classes=len(signs))
    model.summary()

    # Train
    print(f"\n[→] Training for up to {EPOCHS} epochs …\n")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Evaluate
    print("\n[→] Evaluating on test set …")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"    Test Loss     : {loss:.4f}")
    print(f"    Test Accuracy : {acc*100:.2f}%")

    y_pred_prob = model.predict(X_test)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(y_test,      axis=1)

    print("\n── Classification Report ──────────────────")
    print(classification_report(y_true, y_pred, target_names=signs))

    # Plots
    plot_training_history(history, MODEL_DIR)
    plot_confusion_matrix(y_true, y_pred, signs, MODEL_DIR)

    print(f"\n[✓] Best model saved  →  {MODEL_PATH}")
    return model, history


if __name__ == "__main__":
    train()
