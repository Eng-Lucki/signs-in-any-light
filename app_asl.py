import os
import json
import time
import difflib
import subprocess
from collections import Counter, deque
import cv2
import pickle
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import dip_preprocess as dp
from asl_core import HAND_CONNECTIONS, extract_features

# Thresholds calibrated on clean frames by dip_robustness_experiment.ipynb (defaults if not run yet)
ADAPTIVE_THRESHOLDS = None
if os.path.exists("results/adaptive_thresholds.json"):
    with open("results/adaptive_thresholds.json") as f:
        ADAPTIVE_THRESHOLDS = json.load(f)

# 1. Asynchronous Windows Speech Synthesizer
def speak_text(text):
    if not text.strip():
        return
    safe = text.replace("'", "''")  # escape quotes for PowerShell
    cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"
    subprocess.Popen(["powershell", "-Command", cmd], creationflags=subprocess.CREATE_NO_WINDOW)

# 2. Load trained KNN model
with open("knn_model.p", "rb") as f:
    model = pickle.load(f)

DICTIONARY = [
    "HELLO", "WORLD", "CAT", "HAPPY", "COMPUTER", "VISION", 
    "PROJECT", "YES", "NO", "GOOD", "BAD", "TEST", "DEMO", 
    "PYTHON", "CODE", "STUDENT", "UNIVERSITY", "HELP", "THANK"
]

def autocorrect(word):
    if not word or len(word) < 2:
        return word
    matches = difflib.get_close_matches(word.upper(), DICTIONARY, n=1, cutoff=0.55)
    return matches[0] if matches else word

# 3. Setup MediaPipe Landmarker
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

HOLD_TIME_THRESHOLD = 1.0
VOTE_WINDOW = 10        # temporal smoothing: look at the last N frame predictions
VOTE_RATIO = 0.6        # a letter must win at least 60% of them to count as stable
pred_history = deque(maxlen=VOTE_WINDOW)
last_detected_char = None
char_hold_start_time = 0
letter_added = False    # True after a letter is typed, until the hand drops or changes sign
current_word = ""
spoken_sentence = ""

view_mode = 0  # 0 = RGB, 1 = Bilateral Filter, 2 = Canny Edges
enhance_on = True  # adaptive DIP pre-processing before hand detection

# --- GUI LAYOUT: video on the left, dashboard panel on the right, key bar below ---
PANEL_W = 340
KEYBAR_H = 30
TH = {**dp.DEFAULT_THRESHOLDS, **(ADAPTIVE_THRESHOLDS or {})}
fps = 0.0
last_frame_time = time.time()


def text(img, s, org, scale=0.5, color=(220, 220, 220), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def meter(img, x, y, label, frac, marks, bad, width=PANEL_W - 40):
    """Quality meter: label, bar filled to `frac` (0..1), white ticks at the thresholds `marks`."""
    color = (0, 90, 255) if bad else (0, 200, 120)
    text(img, label, (x, y), 0.42, color if bad else (200, 200, 200))
    by = y + 6
    cv2.rectangle(img, (x, by), (x + width, by + 6), (60, 60, 60), -1)
    fill = int(width * min(max(frac, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, by), (x + fill, by + 6), color, -1)
    for m in marks:
        mx = x + int(width * min(max(m, 0.0), 1.0))
        cv2.line(img, (mx, by - 3), (mx, by + 9), (255, 255, 255), 1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    raw = frame.copy()

    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - last_frame_time, 1e-6)  # smoothed frames per second
    last_frame_time = now

    # --- ADAPTIVE PRE-PROCESSING (real-time variant: bilateral instead of NLM) ---
    applied = []
    if enhance_on:
        frame, steps = dp.adaptive_enhance(frame, return_steps=True,
                                           thresholds=ADAPTIVE_THRESHOLDS, denoiser="bilateral")
        applied = [name.split(" (")[0].replace(" denoise", "").replace("auto ", "") for name, _ in steps[1:]]

    # --- CLASSICAL DIP VIEWS (only computed when selected) ---
    if view_mode == 0:
        display_frame = frame.copy()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered_gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        if view_mode == 1:
            display_frame = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)
        else:
            edges = cv2.Canny(filtered_gray, threshold1=100, threshold2=200)
            display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # --- MEDIAPIPE & CLASSIFICATION ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    current_prediction = None
    confidence = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for s, e in HAND_CONNECTIONS:
            cv2.line(display_frame, coords[s], coords[e], (0, 255, 0), 2)
        for pt in coords:
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

        features = extract_features(hand)
        probs = model.predict_proba([features])[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        if confidence > 0.60:
            current_prediction = model.classes_[best_idx]

    # --- TEMPORAL SMOOTHING: majority vote over recent frames ---
    pred_history.append(current_prediction)
    stable_char, votes = Counter(pred_history).most_common(1)[0]
    if votes < VOTE_RATIO * VOTE_WINDOW:
        stable_char = None

    current_time = time.time()
    progress = 0.0

    # A new stable sign (or no sign) restarts the hold timer and re-arms typing.
    if stable_char != last_detected_char:
        last_detected_char = stable_char
        char_hold_start_time = current_time
        letter_added = False

    if stable_char is not None:
        if letter_added:
            progress = 1.0  # already typed: drop the hand or change sign to type again
        else:
            elapsed = current_time - char_hold_start_time
            progress = min(1.0, elapsed / HOLD_TIME_THRESHOLD)
            if elapsed >= HOLD_TIME_THRESHOLD:
                current_word += stable_char
                letter_added = True

    # --- VIDEO OVERLAY: view mode + FPS ---
    mode_str = ["RGB", "Bilateral Filter", "Canny Edges"][view_mode]
    cv2.rectangle(display_frame, (0, 0), (250, 30), (20, 20, 20), -1)
    text(display_frame, f"View: {mode_str}   FPS: {fps:.0f}", (10, 20), 0.5, (0, 255, 255))

    # --- SIDE PANEL DASHBOARD ---
    canvas = np.zeros((h + KEYBAR_H, w + PANEL_W, 3), np.uint8)
    canvas[:h, :w] = display_frame
    panel = canvas[:h, w:]
    panel[:] = (22, 22, 22)
    x, bar_w = 20, PANEL_W - 40

    # Recognized letter: big stable letter, raw per-frame guess beside it
    text(panel, stable_char if stable_char else "-", (x, 72), 2.2, (0, 255, 180), 4)
    text(panel, f"guess: {current_prediction if current_prediction else 'none'} ({int(confidence * 100)}%)",
         (x + 80, 42), 0.5)
    text(panel, f"stable over last {VOTE_WINDOW} frames", (x + 80, 64), 0.4, (140, 140, 140))

    # Hold progress bar (grey once the letter is typed)
    cv2.rectangle(panel, (x, 86), (x + bar_w, 96), (60, 60, 60), -1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        bar_color = (120, 120, 120) if letter_added else (0, 255, 0)
        cv2.rectangle(panel, (x, 86), (x + fill_w, 96), bar_color, -1)
    if letter_added:
        text(panel, "typed - lower hand to repeat", (x, 114), 0.42, (160, 160, 160))

    # Word, autocorrect suggestion, sentence history
    suggested = autocorrect(current_word)
    text(panel, f"Word: {current_word}", (x, 142), 0.7, (255, 255, 255), 2)
    if current_word and suggested != current_word:
        text(panel, f"suggestion: {suggested}", (x, 164), 0.45, (0, 220, 255))
    history = spoken_sentence[-28:] if spoken_sentence else "(empty)"
    text(panel, f"History: {history}", (x, 186), 0.45, (170, 170, 170))
    cv2.line(panel, (x, 198), (PANEL_W - 20, 198), (70, 70, 70), 1)

    # Image-processing section: what was applied, raw vs enhanced thumbnails
    text(panel, "IMAGE PROCESSING", (x, 218), 0.5, (0, 255, 255))
    enh_str = (", ".join(applied) if applied else "none needed") if enhance_on else "OFF  (press 4)"
    text(panel, f"applied: {enh_str}", (x, 237), 0.42, (0, 255, 255))
    tw = (PANEL_W - 50) // 2
    th = int(tw * h / w)
    for i, (img, label) in enumerate([(raw, "raw"), (frame, "enhanced")]):
        tx = x + i * (tw + 10)
        panel[247:247 + th, tx:tx + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (tx, 247 + th - 18), (tx + 70, 247 + th), (20, 20, 20), -1)
        text(panel, label, (tx + 4, 247 + th - 5), 0.4)

    # Live quality meters, measured on the raw frame (what the adaptive pipeline decides on)
    y = 247 + th + 20
    b, n = dp.brightness(raw), dp.noise_sigma(raw)
    c, sh = dp.rms_contrast(raw), dp.sharpness(raw)
    n_max = max(20.0, 3 * TH["noise"])
    sh_max = np.log10(1 + 10 * TH["sharpness"])
    meter(panel, x, y, f"brightness {b:.2f}", b, [TH["dark"], TH["bright"]],
          b < TH["dark"] or b > TH["bright"])
    meter(panel, x, y + 26, f"noise sigma {n:.1f}", n / n_max, [TH["noise"] / n_max], n > TH["noise"])
    meter(panel, x, y + 52, f"contrast {c:.2f}", c / 0.4, [TH["contrast"] / 0.4], c < TH["contrast"])
    meter(panel, x, y + 78, f"sharpness {sh:.0f}", np.log10(1 + sh) / sh_max,
          [np.log10(1 + TH["sharpness"]) / sh_max], sh < TH["sharpness"])

    # Key bar across the bottom
    canvas[h:, :] = (35, 35, 35)
    text(canvas, "[1] RGB  [2] Bilateral  [3] Canny  [4] Enhance on/off  [ENTER] Speak  "
                 "[SPACE] Next word  [BKSP] Delete  [C] Clear  [Q] Quit",
         (10, h + 20), 0.45, (230, 230, 230))

    cv2.imshow("ASL Assistive Communicator", canvas)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        view_mode = 0
    elif key == ord('2'):
        view_mode = 1
    elif key == ord('3'):
        view_mode = 2
    elif key == ord('4'):
        enhance_on = not enhance_on
    elif key == 13:  # ENTER -> Speak
        if current_word:
            final_word = autocorrect(current_word)
            spoken_sentence += final_word + " "
            speak_text(final_word)
            current_word = ""
    elif key == 32:  # SPACE
        if current_word:
            final_word = autocorrect(current_word)
            spoken_sentence += final_word + " "
            current_word = ""
    elif key == 8:   # BACKSPACE
        current_word = current_word[:-1]
    elif key == ord('c') or key == ord('C'):
        current_word = ""
        spoken_sentence = ""

cap.release()
detector.close()
cv2.destroyAllWindows()