# 🤟 Sign Language Detection

A real-time sign language recognition system using **LSTM + MediaPipe + TensorFlow/Keras**.  
Upload a video or use your webcam — the model reads hand/body keypoints and predicts the sign word being shown.

---

## 📁 Project Structure

```
sign_language_detection/
│
├── config.py                  ← All settings (signs, paths, model params)
│
├── utils/
│   └── keypoint_utils.py      ← MediaPipe extraction, drawing helpers
│
├── data_collection/
│   ├── collect_data.py        ← Step 1: Record training data via webcam
│   └── preprocess.py          ← Step 2: Convert .npy frames → X, y tensors
│
├── model/
│   ├── model.py               ← LSTM architecture definition
│   ├── train.py               ← Step 3: Train + evaluate + save model
│   └── evaluate.py            ← Detailed evaluation & plots
│
├── predict_realtime.py        ← Step 4: Live webcam inference
├── app/
│   └── streamlit_app.py       ← Web app (upload video or snapshot)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Signs

Edit `config.py` and set the words you want to teach the model:

```python
SIGNS = [
    "hello", "thanks", "yes", "no", "please",
    "help",  "sorry",  "name", "good", "bad",
]
```

> **Tip:** Start with 5–6 clear, visually distinct signs for best accuracy.

### 3. Collect Training Data

```bash
python data_collection/collect_data.py
```

- Opens your webcam
- For each sign, follow the on-screen countdown and perform the sign
- **30 sequences × 30 frames** per sign are recorded automatically
- Press `Q` to quit early

### 4. Preprocess Data

```bash
python data_collection/preprocess.py
```

Converts raw `.npy` keypoint files → `model/X.npy`, `model/y.npy`.

### 5. Train the Model

```bash
python model/train.py
```

- Trains for up to 200 epochs (early stopping enabled)
- Saves best model to `model/sign_language_model.h5`
- Generates `training_history.png` and `confusion_matrix.png`

### 6. Real-Time Inference

```bash
python predict_realtime.py
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Clear sentence history |

### 7. Web App (optional)

```bash
streamlit run app/streamlit_app.py
```

Opens a browser UI where you can:
- Upload a `.mp4` / `.avi` video for batch inference
- Take a webcam snapshot for quick testing

---

## 🏗️ Model Architecture

```
Input: (batch, 30 frames, 258 keypoints)
          ↓
    LSTM(64, return_seq=True)
          ↓
    LSTM(128, return_seq=True)
          ↓
    LSTM(64)
          ↓
    Dense(64) → BatchNorm → Dropout(0.3)
          ↓
    Dense(32) → BatchNorm → Dropout(0.3)
          ↓
    Dense(num_classes) + Softmax
```

**Keypoint breakdown per frame (258 values):**

| Source | Landmarks | Values each | Total |
|--------|-----------|-------------|-------|
| Pose   | 33        | 4 (x,y,z,vis) | 132 |
| Left hand | 21  | 3 (x,y,z)   | 63  |
| Right hand | 21 | 3 (x,y,z)  | 63  |

---

## ⚙️ Configuration Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIGNS` | 10 words | Sign classes to train |
| `NUM_SEQUENCES` | 30 | Training videos per sign |
| `SEQUENCE_LENGTH` | 30 | Frames per video |
| `LSTM_UNITS` | [64,128,64] | LSTM layer sizes |
| `DENSE_UNITS` | [64,32] | Dense layer sizes |
| `DROPOUT_RATE` | 0.3 | Dropout regularisation |
| `LEARNING_RATE` | 1e-3 | Adam LR |
| `EPOCHS` | 200 | Max training epochs |
| `PREDICTION_THRESHOLD` | 0.85 | Min confidence to display |
| `SMOOTHING_WINDOW` | 5 | Frame smoothing window |

---

## 💡 Tips for Better Accuracy

1. **Record in good lighting** — MediaPipe needs clear hand visibility
2. **Vary your position** — move slightly between sequences
3. **Keep background simple** — reduces noise
4. **Record 50+ sequences** per sign for production use
5. **Add data augmentation** — flip, slight rotation during training

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam not opening | Check `cv2.VideoCapture(0)` → try index 1 or 2 |
| Low accuracy | Collect more data, retrain longer |
| MediaPipe not detecting hands | Improve lighting, keep hands in frame |
| Model not found | Run `train.py` before `predict_realtime.py` |

---

## 🎯 Potential Extensions

- [ ] Add **Indian Sign Language (ISL)** dataset
- [ ] **Text-to-Speech** output with `pyttsx3`
- [ ] **Flask API** for mobile integration
- [ ] **Transformer** backbone instead of LSTM
- [ ] **Word-level** to **sentence-level** language model

---

## 📦 Tech Stack

| Library | Version | Use |
|---------|---------|-----|
| TensorFlow | ≥2.10 | Deep learning |
| Keras | ≥2.10 | Model API |
| MediaPipe | ≥0.10 | Keypoint extraction |
| OpenCV | ≥4.7 | Video I/O |
| NumPy | ≥1.23 | Array ops |
| Streamlit | ≥1.22 | Web UI |
| scikit-learn | ≥1.2 | Metrics, splits |
| Matplotlib/Seaborn | — | Plots |
