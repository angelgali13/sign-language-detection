import cv2, numpy as np, mediapipe as mp, urllib.request, os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand model (~8MB)...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", MODEL_PATH)
    print("Done.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=VisionRunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
_landmarker = HandLandmarker.create_from_options(options)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

def mediapipe_detection(frame, model=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = _landmarker.detect(mp_image)
    return frame, results

def draw_styled_landmarks(frame, results):
    if not results.hand_landmarks: return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    colors = [(0,140,255),(200,0,200)]
    for idx, hand in enumerate(results.hand_landmarks):
        c = colors[idx % 2]
        for s,e in CONNECTIONS:
            cv2.line(overlay,(int(hand[s].x*w),int(hand[s].y*h)),(int(hand[e].x*w),int(hand[e].y*h)),c,2)
        for lm in hand:
            cv2.circle(overlay,(int(lm.x*w),int(lm.y*h)),4,c,-1)
    return cv2.addWeighted(overlay,0.8,frame,0.2,0)

def extract_keypoints(results):
    lh = np.zeros(63); rh = np.zeros(63)
    if results.hand_landmarks and results.handedness:
        for hand, handedness in zip(results.hand_landmarks, results.handedness):
            coords = np.array([[lm.x,lm.y,lm.z] for lm in hand]).flatten()
            if handedness[0].category_name == 'Left': lh = coords
            else: rh = coords
    return np.concatenate([lh, rh])

def put_text_with_bg(frame, text, pos, font_scale=1.0, color=(255,255,255), bg_color=(0,0,0), thickness=2, padding=8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),baseline = cv2.getTextSize(text,font,font_scale,thickness)
    x,y = pos
    cv2.rectangle(frame,(x-padding,y-th-padding),(x+tw+padding,y+baseline+padding),bg_color,-1)
    cv2.putText(frame,text,(x,y),font,font_scale,color,thickness,cv2.LINE_AA)
    return frame

def draw_prediction_bar(frame, predictions, signs, top_k=5):
    h,w = frame.shape[:2]; bar_x = w-220
    for i,idx in enumerate(np.argsort(predictions)[::-1][:top_k]):
        y = 30+i*40
        cv2.rectangle(frame,(bar_x,y-15),(bar_x+int(predictions[idx]*190),y+5),(0,255,0) if i==0 else (0,180,180),-1)
        cv2.putText(frame,f"{signs[idx]}: {predictions[idx]:.2f}",(bar_x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    return frame
