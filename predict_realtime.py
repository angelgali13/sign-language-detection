#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  Sign Language Detection — Real-Time Inference
#
#  Usage:
#      python predict_realtime.py
#
#  Press  Q  to quit.
# ─────────────────────────────────────────────

import os
import sys
import collections

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (MODEL_PATH, LABEL_PATH, SEQUENCE_LENGTH,
                    PREDICTION_THRESHOLD, SMOOTHING_WINDOW)
from utils.keypoint_utils import (mediapipe_detection,
                                   draw_styled_landmarks,
                                   extract_keypoints,
                                   draw_prediction_bar,
                                   put_text_with_bg)

mp_holistic = mp.solutions.holistic


# ── Load model & labels ───────────────────────
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Train the model first:  python model/train.py"
        )
    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = list(np.load(LABEL_PATH, allow_pickle=True))
    print(f"[✓] Model loaded  ({len(labels)} classes: {labels})")
    return model, labels


# ── Prediction smoother ───────────────────────
class PredictionSmoother:
    """
    Keeps a rolling window of raw predictions and returns
    the class with the highest mean probability, but only
    if it exceeds the confidence threshold.
    """
    def __init__(self, window=SMOOTHING_WINDOW, threshold=PREDICTION_THRESHOLD):
        self.window    = collections.deque(maxlen=window)
        self.threshold = threshold

    def update(self, probs):
        self.window.append(probs)

    def get_prediction(self):
        if len(self.window) < self.window.maxlen:
            return None, 0.0
        avg   = np.mean(self.window, axis=0)
        idx   = int(np.argmax(avg))
        conf  = float(avg[idx])
        return (idx, conf) if conf >= self.threshold else (None, conf)


# ── HUD drawing ───────────────────────────────
def draw_hud(frame, sentence, current_sign, confidence,
             predictions, labels, frame_count, seq_len):
    h, w = frame.shape[:2]

    # Prediction bar (top-right)
    if predictions is not None:
        draw_prediction_bar(frame, predictions, labels, top_k=min(5, len(labels)))

    # Current prediction banner
    if current_sign:
        put_text_with_bg(frame, f"► {current_sign.upper()}  ({confidence:.0%})",
                         (20, 60), font_scale=1.4,
                         color=(0, 255, 100), bg_color=(0, 50, 0),
                         thickness=2, padding=12)

    # Sentence history (bottom)
    sentence_str = "  ".join(sentence[-8:])
    put_text_with_bg(frame, sentence_str or "Waiting for sign…",
                     (20, h - 20), font_scale=0.8,
                     color=(220, 220, 220), bg_color=(30, 30, 30),
                     padding=8)

    # Frame buffer progress bar
    prog = int((frame_count / seq_len) * (w // 3))
    cv2.rectangle(frame, (20, h - 45), (20 + prog, h - 38),
                  (0, 180, 255), -1)
    cv2.rectangle(frame, (20, h - 45), (20 + (w // 3), h - 38),
                  (80, 80, 80), 1)

    # Controls hint
    cv2.putText(frame, "Q: quit  |  C: clear sentence",
                (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


# ── Main loop ─────────────────────────────────
def run():
    model, labels = load_artifacts()
    smoother      = PredictionSmoother()

    sequence  = []       # rolling frame buffer
    sentence  = []       # confirmed sign history
    last_sign = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[→] Real-time inference running. Press Q to quit.\n")

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            _, results = mediapipe_detection(frame, holistic)
            frame = draw_styled_landmarks(frame, results)

            # Build keypoint sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            predictions = None
            current_sign, confidence = None, 0.0

            if len(sequence) == SEQUENCE_LENGTH:
                X          = np.expand_dims(sequence, axis=0).astype(np.float32)
                predictions = model.predict(X, verbose=0)[0]
                smoother.update(predictions)
                pred_idx, confidence = smoother.get_prediction()

                if pred_idx is not None:
                    current_sign = labels[pred_idx]
                    # Append to sentence only when sign changes
                    if current_sign != last_sign:
                        sentence.append(current_sign)
                        last_sign = current_sign

            frame = draw_hud(frame, sentence, current_sign,
                             confidence, predictions, labels,
                             len(sequence), SEQUENCE_LENGTH)

            cv2.imshow("Sign Language Detection  —  Real-Time", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()
                last_sign = None
                print("[i] Sentence cleared")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] Session ended.")
    if sentence:
        print(f"    Final sentence: {' '.join(sentence)}")


if __name__ == "__main__":
    run()
