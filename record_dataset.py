"""Record a raw-image ASL dataset for the robustness experiment.

Unlike train_knn.py (which only keeps landmark features), this saves the actual
frames, so the notebook can degrade and enhance them later.

Usage:
    python record_dataset.py --session alice_day
    python record_dataset.py --session bob_evening --samples 30

Record at least two sessions (different people and/or rooms/lighting): the
notebook trains on some sessions and tests on another, so test images are
never near-duplicates of training images.

Layout:  dataset/<session>/<letter>/<letter>_<nnn>.jpg
"""
import argparse
import os
import cv2

from asl_core import CLASSES, make_detector, detect_landmarks

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True, help="e.g. alice_day, bob_lamp")
parser.add_argument("--samples", type=int, default=30, help="images per letter")
parser.add_argument("--every", type=int, default=4, help="save every Nth detected frame")
parser.add_argument("--out", default="dataset")
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()

detector = make_detector(min_conf=0.5)
cap = cv2.VideoCapture(args.camera)

print(f"\n--- Recording session '{args.session}' ({len(CLASSES)} letters x {args.samples}) ---")
print("Press 's' to start a letter, 'k' to skip it, 'q' to quit.")
print("While recording, move and rotate your hand slightly and vary the distance.\n")

quit_all = False
for letter in CLASSES:
    folder = os.path.join(args.out, args.session, letter)
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
    if existing >= args.samples:
        print(f"{letter}: already has {existing} images, skipping")
        continue

    skip = False
    while True:
        ret, frame = cap.read()
        if not ret:
            quit_all = True
            break
        frame = cv2.flip(frame, 1)
        view = frame.copy()
        cv2.putText(view, f"Ready for letter: {letter}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(view, "[s] start  [k] skip  [q] quit", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dataset recorder", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        if key == ord("k"):
            skip = True
            break
        if key == ord("q"):
            quit_all = True
            break
    if quit_all:
        break
    if skip:
        continue

    count, seen = existing, 0
    while count < args.samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        view = frame.copy()
        # Only keep frames where the hand is visible, so 'clean' is a fair reference.
        if detect_landmarks(detector, frame) is not None:
            seen += 1
            if seen % args.every == 0:
                count += 1
                cv2.imwrite(os.path.join(folder, f"{letter}_{count:03d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.putText(view, f"Recording {letter}: {count}/{args.samples}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Dataset recorder", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            quit_all = True
            break
    if quit_all:
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
print("Done. Images saved under", os.path.join(args.out, args.session))
