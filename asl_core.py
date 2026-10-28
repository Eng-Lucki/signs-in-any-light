"""Shared hand-landmark helpers used by the recorder and the robustness notebook.

extract_features() is identical to the one in train_knn.py / app_asl.py so that
results from the notebook transfer directly to the live app.
"""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# 24 static ASL letters. J and Z need motion, so a single frame cannot capture them.
CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def make_detector(min_conf=0.5, model_path=MODEL_PATH):
    """Hand landmarker in IMAGE mode (every frame is independent, no tracking)."""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_landmarks(detector, bgr):
    """Return a (21, 3) array of normalized landmarks, or None if no hand is found."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])


def extract_features(coords):
    """93-D feature vector: wrist-relative, scale-normalized coords + fingertip distances.

    Accepts either a (21, 3) array or a MediaPipe landmark list.
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in coords])
    wrist = coords[0]
    scale = np.linalg.norm(coords[9] - coords[0])
    if scale < 1e-6:
        scale = 1.0
    rel_coords = ((coords - wrist) / scale).flatten()

    fingertip_ids = [4, 8, 12, 16, 20]
    knuckle_ids = [0, 2, 5, 9, 13, 17]
    distances = []
    for tip in fingertip_ids:
        distances.append(np.linalg.norm(coords[tip] - wrist) / scale)
        for kn in knuckle_ids:
            if kn != 0:
                distances.append(np.linalg.norm(coords[tip] - coords[kn]) / scale)

    return np.concatenate([rel_coords, distances])


def landmark_error(coords, ref_coords):
    """Mean 2-D landmark displacement, in units of hand size (wrist -> middle knuckle)."""
    scale = np.linalg.norm(ref_coords[9, :2] - ref_coords[0, :2])
    if scale < 1e-6:
        scale = 1.0
    return float(np.mean(np.linalg.norm(coords[:, :2] - ref_coords[:, :2], axis=1)) / scale)
