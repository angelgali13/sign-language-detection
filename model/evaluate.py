#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  Sign Language Detection — Evaluation Script
#
#  Loads the saved model, runs on the test split,
#  prints metrics, and saves visual plots.
#
#  Usage:
#      python model/evaluate.py
# ─────────────────────────────────────────────

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_DIR, MODEL_PATH, VALIDATION_SPLIT


def evaluate():
    # Load artifacts
    for p in [MODEL_PATH,
              os.path.join(MODEL_DIR, "X.npy"),
              os.path.join(MODEL_DIR, "y.npy"),
              os.path.join(MODEL_DIR, "labels.npy")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    model  = tf.keras.models.load_model(MODEL_PATH)
    X      = np.load(os.path.join(MODEL_DIR, "X.npy"))
    y      = np.load(os.path.join(MODEL_DIR, "y.npy"))
    labels = list(np.load(os.path.join(MODEL_DIR, "labels.npy"),
                          allow_pickle=True))

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=np.argmax(y, axis=1),
    )

    # Metrics
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'─'*45}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"{'─'*45}\n")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(y_test,      axis=1)

    print(classification_report(y_true, y_pred, target_names=labels))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "confusion_matrix_eval.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Confusion matrix saved  →  {path}")
    plt.close()

    # Per-class accuracy
    per_class = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 5))
    colors = ["#4CAF50" if v >= 0.8 else "#FF9800" if v >= 0.6 else "#F44336"
              for v in per_class]
    bars = plt.bar(labels, per_class * 100, color=colors)
    plt.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% target")
    plt.title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    for bar, val in zip(bars, per_class):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{val*100:.0f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Per-class accuracy saved  →  {path}")
    plt.close()


if __name__ == "__main__":
    evaluate()
