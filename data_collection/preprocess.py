#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  Sign Language Detection — Preprocessing
#
#  Loads saved .npy keypoint files → builds
#  (X, y) tensors → saves them for training.
# ─────────────────────────────────────────────

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (DATASET_DIR, MODEL_DIR, SIGNS,
                    NUM_SEQUENCES, SEQUENCE_LENGTH, KEYPOINT_SIZE)


def load_dataset():
    """
    Walk dataset/<sign>/<seq>/<frame>.npy and build:
        X  : (N, SEQUENCE_LENGTH, KEYPOINT_SIZE)  float32
        y  : (N, num_classes)                     one-hot
    """
    label_map = {sign: idx for idx, sign in enumerate(SIGNS)}
    print(f"[i] Labels: {label_map}\n")

    sequences, labels = [], []
    missing = 0

    for sign in SIGNS:
        for seq in range(NUM_SEQUENCES):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                path = os.path.join(DATASET_DIR, sign,
                                    str(seq), f"{frame_num}.npy")
                if not os.path.exists(path):
                    missing += 1
                    window.append(np.zeros(KEYPOINT_SIZE))
                else:
                    window.append(np.load(path))

            sequences.append(window)
            labels.append(label_map[sign])

    X = np.array(sequences, dtype=np.float32)   # (N, T, F)
    y = to_categorical(labels, num_classes=len(SIGNS))

    print(f"[✓] Dataset loaded")
    print(f"    X shape : {X.shape}")
    print(f"    y shape : {y.shape}")
    if missing:
        print(f"    [!] {missing} missing frames replaced with zeros")

    return X, y, SIGNS


def save_processed(X, y):
    np.save(os.path.join(MODEL_DIR, "X.npy"), X)
    np.save(os.path.join(MODEL_DIR, "y.npy"), y)
    np.save(os.path.join(MODEL_DIR, "labels.npy"), np.array(SIGNS))
    print(f"[✓] Saved  X.npy, y.npy, labels.npy  →  {MODEL_DIR}")


def get_splits(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state,
                            stratify=np.argmax(y, axis=1))


if __name__ == "__main__":
    X, y, signs = load_dataset()
    save_processed(X, y)

    X_train, X_test, y_train, y_test = get_splits(X, y)
    print(f"\n    Train : {X_train.shape[0]} samples")
    print(f"    Test  : {X_test.shape[0]} samples")
