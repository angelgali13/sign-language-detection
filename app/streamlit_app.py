import os, sys
import streamlit as st
import numpy as np
import pickle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Sign Language Detector", page_icon="🤟", layout="wide")

st.markdown("""
<style>
.main-title { font-size: 2.8rem; font-weight: 800; color: #4CAF50; text-align: center; }
.sub-title  { font-size: 1rem; color: #aaa; text-align: center; margin-bottom: 2rem; }
.pred-box   { background: #1e3a1e; border: 2px solid #4CAF50; border-radius: 12px;
              padding: 1.5rem; text-align: center; font-size: 2.5rem;
              font-weight: bold; color: #4CAF50; margin: 1rem 0; }
.stat-box   { background: #1a1a2e; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤟 Sign Language Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time sign language recognition using AI</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    path = 'model/rf_model.pkl'
    if not os.path.exists(path):
        return None, []
    data = pickle.load(open(path, 'rb'))
    return data['model'], list(data['labels'])

model, labels = load_model()

if model is None:
    st.error("❌ Model not found!")
    st.stop()

st.success(f"✅ Model loaded — {len(labels)} signs: **{', '.join(labels)}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-box"><h3>🎯 96.67%</h3><p>Model Accuracy</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-box"><h3>🤟 {len(labels)}</h3><p>Signs Supported</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><h3>⚡ Real-Time</h3><p>Live Detection</p></div>', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🤟 Supported Signs", "📊 Model Info", "ℹ️ About"])

with tab1:
    st.subheader("10 Supported Signs")
    st.info("💡 For live detection run locally: `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python predict_realtime.py`")

    signs_info = {
        "hello":  ("👋", "Open palm, wave from forehead outward like a salute"),
        "thanks": ("🙏", "Flat hand touches chin/lips, move forward"),
        "yes":    ("✊", "Make a fist, nod it up and down"),
        "no":     ("✌️", "Index + middle finger, wave side to side"),
        "please": ("🙏", "Flat hand on chest, rub in a circle"),
        "help":   ("👍", "Thumbs up on open palm, lift both up"),
        "sorry":  ("✊", "Fist on chest, rub in a circle"),
        "name":   ("👆", "Two fingers tap on top of two fingers"),
        "good":   ("👍", "Flat hand from chin, move forward and down"),
        "bad":    ("👎", "Flat hand from chin, flip hand downward"),
    }

    cols = st.columns(2)
    for i, (sign, (emoji, desc)) in enumerate(signs_info.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:#1a1a2e;border-radius:8px;padding:1rem;margin:0.5rem 0;border-left:4px solid #4CAF50'>
                <h3>{emoji} {sign.upper()}</h3>
                <p style='color:#aaa'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Model Performance")
    st.markdown("### 📊 Classification Report")

    report_data = {
        "Sign":      ["hello","thanks","yes","no","please","help","sorry","name","good","bad"],
        "Precision": [1.00, 0.83, 1.00, 1.00, 1.00, 0.83, 1.00, 1.00, 1.00, 1.00],
        "Recall":    [1.00, 1.00, 0.86, 1.00, 1.00, 1.00, 0.80, 1.00, 1.00, 1.00],
        "F1-Score":  [1.00, 0.91, 0.92, 1.00, 1.00, 0.91, 0.89, 1.00, 1.00, 1.00],
    }
    import pandas as pd
    df = pd.DataFrame(report_data)
    st.dataframe(df, use_container_width=True)
    st.metric("Overall Accuracy", "96.67%")

    st.markdown("### 🏗️ Architecture")
    st.code("""
Input: Video frames
    ↓
MediaPipe Hand Landmarker
    ↓
126 keypoints per frame (2 hands × 21 × 3)
    ↓
30 frames sequence → flatten → 3780 features
    ↓
Random Forest (200 trees)
    ↓
Predicted Sign + Confidence
    """)

with tab3:
    st.markdown("""
## About This Project

A **real-time sign language detection** system that helps deaf people communicate.

### Tech Stack
| Component | Technology |
|---|---|
| Hand Detection | MediaPipe Tasks API |
| Classification | Random Forest (sklearn) |
| Video Processing | OpenCV |
| Web App | Streamlit |

### How to run locally
```bash
conda activate signlang
cd sign_language_detection
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python predict_realtime.py
```

### Dataset
- 10 signs × 30 sequences × 30 frames = 9,000 frames
- 126 keypoints per frame
- Train/Test split: 80/20
    """)
