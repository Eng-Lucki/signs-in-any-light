# Signs in Any Light

Real-time American Sign Language (ASL) fingerspelling to text and speech, made robust to bad capture conditions with classical image processing.

Digital Image Processing course project, College of Computer Engineering, University of Technology – Iraq.

## The idea

Sign recognition usually works well in good lighting, but people often sign in dark rooms, in front of bright windows, or with cheap noisy webcams. In those conditions the hand detector loses the hand or puts the landmarks in the wrong place, and the translation fails.

Our system adds an image-processing stage **before** the hand is detected. It measures each frame (brightness, noise, contrast, sharpness) and applies only the corrections that frame needs: denoising, gamma correction, CLAHE or sharpening.

```
webcam → measure quality → adaptive enhancement → MediaPipe hand landmarks → features → KNN → text + speech
```

## Features

- Recognizes the 24 static ASL letters (J and Z need motion, so they are not included)
- Adaptive enhancement: non-local means or bilateral denoising, gamma correction, CLAHE, unsharp masking
- Live app with raw vs. enhanced view, quality meters, FPS counter and an enhancement on/off toggle
- Temporal smoothing, so a letter is typed only when it is stable, and holding a sign does not repeat it
- Autocorrect and text-to-speech (Windows)
- Experiment notebook that degrades images on purpose (low light, noise, blur, backlight, JPEG) and compares enhancement methods

## Project structure

| File | What it does |
| --- | --- |
| `app_asl.py` | Live webcam translator with the dashboard |
| `train_knn.py` | Trains the KNN classifier from the recorded images (`knn_model.p`) |
| `record_dataset.py` | Records hand images per letter (used for training and the experiment) |
| `dip_preprocess.py` | Degradations, enhancement methods, adaptive pipeline and quality metrics |
| `asl_core.py` | Shared helpers: letters, hand detector, landmark features |
| `dip_robustness_experiment.ipynb` | Robustness experiment: degradation × method, parameter studies, figures |
| `hand_landmarker.task` | MediaPipe hand landmark model |

## Setup

```bash
pip install -r requirements.txt
```

Run every command from the project folder.

## Usage

1. **Record your hand images** (webcam needed). For each letter, press `s` to start recording. Record at least two sessions, ideally different people or rooms:
   ```bash
   python record_dataset.py --session person1_day
   python record_dataset.py --session person2_evening
   ```
   Images are saved in `dataset/<session>/<letter>/`.

2. **Train the classifier** from those images (no webcam needed):
   ```bash
   python train_knn.py
   ```
   It prints the accuracy on a session the model has never seen and saves `knn_model.p`.

3. **Run the app:**
   ```bash
   python app_asl.py
   ```
   | Key | Action |
   | --- | --- |
   | `1` / `2` / `3` | Normal / bilateral / Canny view |
   | `4` | Enhancement on/off |
   | `ENTER` | Speak the word |
   | `SPACE` | Next word |
   | `BACKSPACE` | Delete the last letter |
   | `C` | Clear |
   | `Q` | Quit |

   Hold a sign until the bar fills to type it. Lower your hand to type the same letter twice.

4. **Run the experiment (optional)** on the same recordings:
   ```bash
   python -m jupyter notebook dip_robustness_experiment.ipynb
   ```
   The notebook saves its figures to `figures/` and its tables to `results/`. It also saves thresholds calibrated to your camera, which the app then uses automatically.

## Notes

- Text-to-speech uses the Windows speech synthesizer, so speech only works on Windows. Everything else runs anywhere.
- `knn_model.p`, `dataset/`, `results/` and `figures/` are created locally and are not committed.
