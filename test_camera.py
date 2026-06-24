#!/usr/bin/env python3
import os, sys
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
if os.environ.get("SLD_CHILD") != "1":
    import subprocess
    e = os.environ.copy(); e["SLD_CHILD"] = "1"
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=e).returncode)
import cv2, mediapipe as mp
print("Starting camera test...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened(): cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CAMERA FAILED - System Settings > Privacy > Camera > allow Terminal/VSCode")
    sys.exit(1)
print("Camera OK - window opening...")
mpH = mp.solutions.holistic; mpD = mp.solutions.drawing_utils
with mpH.Holistic(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        r = h.process(rgb); rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if r.right_hand_landmarks: mpD.draw_landmarks(frame, r.right_hand_landmarks, mpH.HAND_CONNECTIONS)
        if r.left_hand_landmarks:  mpD.draw_landmarks(frame, r.left_hand_landmarks,  mpH.HAND_CONNECTIONS)
        hands = ("Right " if r.right_hand_landmarks else "") + ("Left" if r.left_hand_landmarks else "")
        cv2.putText(frame, f"Hands: {hands or 'None'}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, "Q to quit", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        cv2.imshow("Camera Test - OK!", frame); n += 1
        if n == 5: print("MediaPipe running OK!")
        if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
print(f"SUCCESS - {n} frames processed. Camera + MediaPipe work!")
