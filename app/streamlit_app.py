import os, sys
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import streamlit as st
import cv2
import numpy as np
import pickle
import collections
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.keypoint_utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# ── Page config ────────────────────────────────
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size: 2.8rem; font-weight: 800; color: #4CAF50; text-align: center; }
.sub-title  { font-size: 1rem; color: #aaa; text-align: center; margin-bottom: 2rem; }
.pred-box   { background: #1e3a1e; border: 2px solid #4CAF50; border-radius: 12px;
              padding: 1.5rem; text-align: center; font-size: 2.5rem;
              font-weight: bold; color: #4CAF50; margin: 1rem 0; }
.sentence-box { background: #1a1a2e; border: 1px solid #555; border-radius: 8px;
                padding: 1rem; font-size: 1.3rem; color: #ddd; min-height: 3rem; }
.stat-box   { background: #1a1a2e; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤟 Sign Language Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time sign language recognition using AI</div>', unsafe_allow_html=True)

# ── Load model ─────────────────────────────────
@st.cache_resource
def load_model():
    path = 'model/rf_model.pkl'
    if not os.path.exists(path):
        return None, []
    data = pickle.load(open(path, 'rb'))
    return data['model'], list(data['labels'])

model, labels = load_model()

if model is None:
    st.error("❌ Model not found. Train the model first!")
    st.stop()

st.success(f"✅ Model loaded — {len(labels)} signs: **{', '.join(labels)}**")

# ── Stats ──────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-box"><h3>🎯 96.67%</h3><p>Model Accuracy</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-box"><h3>🤟 {len(labels)}</h3><p>Signs Supported</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><h3>⚡ Real-Time</h3><p>Live Detection</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ───────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Snapshot Detection", "🎬 Video Upload", "ℹ️ About"])

# ── Tab 1: Snapshot ────────────────────────────
with tab1:
    st.subheader("Take a snapshot and detect the sign")
    st.info("💡 For live webcam detection, run: `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python predict_realtime.py`")

    snapshot = st.camera_input("📷 Show your sign and take a photo")

    if snapshot:
        img_bytes = snapshot.getvalue()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            _, results = mediapipe_detection(frame)
            frame_drawn = draw_styled_landmarks(frame.copy(), results)
            st.image(cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB),
                     caption="Hand landmarks detected", use_column_width=True)

        with col2:
            kp = extract_keypoints(results)
            # Repeat single frame to fill sequence
            sequence = [kp] * 30
            flat = np.array(sequence).flatten().reshape(1, -1)
            probs = model.predict_proba(flat)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            sign = labels[idx]

            if conf > 0.5:
                st.markdown(f'<div class="pred-box">🤟 {sign.upper()}<br><span style="font-size:1rem;color:#aaa">{conf:.0%} confidence</span></div>', unsafe_allow_html=True)
            else:
                st.warning(f"Low confidence ({conf:.0%}) — try a clearer sign")

            st.markdown("**All sign probabilities:**")
            chart_data = dict(zip(labels, [float(p) for p in probs]))
            st.bar_chart(chart_data)

# ── Tab 2: Video Upload ────────────────────────
with tab2:
    st.subheader("Upload a video to detect signs")
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if video_file:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.video(video_file)

        with st.spinner("Analysing video..."):
            cap = cv2.VideoCapture(tmp_path)
            sequence = []; sentence = []; last_sign = None
            window = collections.deque(maxlen=5)
            frame_count = 0

            progress = st.progress(0, text="Processing...")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                _, results = mediapipe_detection(frame)
                kp = extract_keypoints(results)
                sequence.append(kp)
                if len(sequence) > 30: sequence.pop(0)
                if len(sequence) == 30:
                    flat = np.array(sequence).flatten().reshape(1, -1)
                    probs = model.predict_proba(flat)[0]
                    idx = int(np.argmax(probs)); conf = float(probs[idx])
                    window.append(idx)
                    if len(window) == 5:
                        final = max(set(window), key=list(window).count)
                        if conf > 0.7:
                            sign = labels[final]
                            if sign != last_sign:
                                sentence.append(sign); last_sign = sign
                frame_count += 1
                if total > 0:
                    progress.progress(min(frame_count/total, 1.0))

            cap.release()
            progress.empty()
            os.unlink(tmp_path)

        st.markdown("**Detected sentence:**")
        result = " → ".join(sentence) if sentence else "No signs detected"
        st.markdown(f'<div class="sentence-box">{result}</div>', unsafe_allow_html=True)

# ── Tab 3: About ───────────────────────────────
with tab3:
    st.markdown("""
## About This Project

A **real-time sign language detection** system built with:

| Component | Technology |
|---|---|
| Hand Detection | MediaPipe Tasks API |
| Classification | Random Forest (sklearn) |
| Video Processing | OpenCV |
| Web App | Streamlit |

### How it works
1. **MediaPipe** extracts 21 hand landmarks (x, y, z) per hand = 126 keypoints per frame
2. **30 frames** are collected as one sequence
3. **Random Forest** classifies the sequence into one of 10 signs
4. Result displayed in real-time

### Supported Signs
""")
    cols = st.columns(5)
    emojis = ["👋","🙏","✊","✌️","🙏","👍","✊","👆","👍","👎"]
    for i, (sign, emoji) in enumerate(zip(labels, emojis)):
        with cols[i % 5]:
            st.markdown(f"**{emoji} {sign.upper()}**")
