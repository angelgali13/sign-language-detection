#!/usr/bin/env python3
import os, sys
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR, SIGNS, NUM_SEQUENCES, SEQUENCE_LENGTH, COLLECTION_WAIT
from utils.keypoint_utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, put_text_with_bg

def collect():
    for sign in SIGNS:
        for seq in range(NUM_SEQUENCES):
            os.makedirs(os.path.join(DATASET_DIR, sign, str(seq)), exist_ok=True)
    print("Folders ready")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Camera failed"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera OK - starting...")
    for si, sign in enumerate(SIGNS):
        print(f"\n--- {sign.upper()} ({si+1}/{len(SIGNS)}) ---")
        for seq in range(NUM_SEQUENCES):
            print(f"  Seq {seq+1}/{NUM_SEQUENCES} GET READY", flush=True)
            for _ in range(COLLECTION_WAIT):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                _, results = mediapipe_detection(frame)
                frame = draw_styled_landmarks(frame, results)
                put_text_with_bg(frame, f"GET READY: {sign.upper()} ({seq+1}/{NUM_SEQUENCES})",
                                 (20,60), font_scale=1.0, color=(0,255,255), bg_color=(0,70,0), padding=10)
                cv2.imshow("Sign Collection", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return
            for fn in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                _, results = mediapipe_detection(frame)
                frame = draw_styled_landmarks(frame, results)
                np.save(os.path.join(DATASET_DIR, sign, str(seq), str(fn)), extract_keypoints(results))
                cv2.circle(frame, (28,28), 10, (0,0,255), -1)
                put_text_with_bg(frame, f"REC {fn+1}/{SEQUENCE_LENGTH}", (45,38),
                                 font_scale=0.65, color=(255,255,255), bg_color=(160,0,0))
                cv2.imshow("Sign Collection", frame)
                cv2.waitKey(1)
            print(f"    seq {seq+1} done", flush=True)
    cap.release(); cv2.destroyAllWindows()
    print("\nDone!")
    for sign in SIGNS:
        count = sum(len(f) for _,_,f in os.walk(os.path.join(DATASET_DIR, sign)) if f)
        print(f"  {sign:<15} -> {count} frames")

if __name__ == "__main__": collect()
