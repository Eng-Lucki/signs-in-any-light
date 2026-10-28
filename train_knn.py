"""Train the KNN letter classifier from the images recorded with record_dataset.py (saves knn_model.p).

No webcam needed: every saved image is run through the hand detector, its 21
landmarks are turned into features, and the classifier is trained on them.

Usage:
    python train_knn.py
    python train_knn.py --dataset dataset --k 3
"""
import argparse
import glob
import os
import pickle

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from asl_core import make_detector, detect_landmarks, extract_features

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset", help="folder created by record_dataset.py")
parser.add_argument("--k", type=int, default=3, help="number of neighbours")
parser.add_argument("--out", default="knn_model.p")
args = parser.parse_args()

paths = sorted(glob.glob(os.path.join(args.dataset, "*", "*", "*.jpg")))
if not paths:
    raise SystemExit(f"No images found in '{args.dataset}/'. Record some first:\n"
                     "    python record_dataset.py --session yourname_day")

detector = make_detector(min_conf=0.5)
X, y, sessions = [], [], []
skipped = 0
for i, path in enumerate(paths, 1):
    session, letter = path.split(os.sep)[-3:-1]
    img = cv2.imread(path)
    landmarks = detect_landmarks(detector, img) if img is not None else None
    if landmarks is None:
        skipped += 1
    else:
        X.append(extract_features(landmarks))
        y.append(letter)
        sessions.append(session)
    if i % 200 == 0 or i == len(paths):
        print(f"Extracting landmarks: {i}/{len(paths)}")
detector.close()

if not X:
    raise SystemExit("No hand was detected in any image - check the recordings.")
X, y, sessions = np.array(X), np.array(y), np.array(sessions)
print(f"{len(X)} samples, {len(set(y))} letters, {len(set(sessions))} session(s); {skipped} images had no hand")


def make_model():
    return KNeighborsClassifier(n_neighbors=args.k, weights="distance")


# Honest accuracy estimate: test on a session the model never saw. The held-out session
# must only contain letters the other sessions cover; the largest such session is used.
# Without one, fall back to a random split (optimistic: near-duplicate frames).
candidates = [s for s in sorted(set(sessions))
              if set(y[sessions == s]) <= set(y[sessions != s])]
if candidates:
    held_out = max(candidates, key=lambda s: int((sessions == s).sum()))
    test = sessions == held_out
    model = make_model().fit(X[~test], y[~test])
    acc = accuracy_score(y[test], model.predict(X[test]))
    print(f"Accuracy on unseen session '{held_out}' ({int(test.sum())} images): {acc * 100:.1f}%")
else:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    acc = accuracy_score(y_te, make_model().fit(X_tr, y_tr).predict(X_te))
    print(f"Accuracy on a random 20% split (single session, likely optimistic): {acc * 100:.1f}%")

# The model used by the app is trained on everything.
model = make_model().fit(X, y)
with open(args.out, "wb") as f:
    pickle.dump(model, f)
print(f"Saved to {args.out}")
