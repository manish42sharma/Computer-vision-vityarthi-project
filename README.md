# Sign Language Recognition System

A fully command-line, computer-vision pipeline that recognises static
sign-language hand gestures (fingerspelling-style signs) from images,
video files, or a live webcam. Built for the Computer Vision course
"Build Your Own Project" evaluation.

Hand poses are located with **MediaPipe Hands** (21 3-D landmarks per
hand), normalised into a 63-length feature vector, and classified with a
classical **scikit-learn** model (Random Forest by default; SVM and MLP
are also selectable). No GPU, no cloud service, and no GUI are required
to run the core pipeline.

---

## 1. Overview

| | |
|---|---|
| **Input** | A photo, a folder of photos, a video file, or a live webcam feed |
| **Output** | Predicted sign label + confidence score (stdout, optionally CSV) |
| **Core technique** | MediaPipe hand-landmark extraction + scikit-learn classifier |
| **Interface** | 100% command line (`python -m src.<module>`) |

## 2. Features

1. **Data Collection & Feature Extraction** (`src/collect_data.py`,
   `src/feature_extraction.py`) — build a labelled dataset of normalised
   hand-landmark vectors from a folder of images or a webcam.
2. **Model Training & Evaluation** (`src/train_model.py`,
   `src/evaluate.py`) — train Random Forest / SVM / MLP classifiers with
   a train/test split and k-fold cross-validation, then produce a full
   classification report and confusion matrix.
3. **Sign Recognition** (`src/predict.py`) — recognise a sign from a
   single image, a folder of images, or a video file, entirely from the
   terminal, with an optional CSV export.
4. *(Optional, bonus)* **Live webcam demo** (`src/live_demo.py`) — an
   interactive on-screen overlay for anyone with a camera and a display.
5. **Synthetic demo dataset generator** (`src/generate_demo_dataset.py`)
   — lets you sanity-check the whole training/evaluation/inference
   pipeline in under a minute, before you have collected any real data.

## 3. Project Structure

```
sign-language-recognition/
├── README.md                  <- you are here
├── statement.md                <- problem statement, scope, target users
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py                Central configuration (paths, thresholds)
│   ├── utils.py                 Logging + error-handling helpers
│   ├── feature_extraction.py    MediaPipe hand-landmark extraction (core CV)
│   ├── collect_data.py          Module 1: Data collection CLI
│   ├── dataset.py               Dataset loading / train-test split
│   ├── model.py                 Model factory + persistence
│   ├── train_model.py           Module 2: Training CLI
│   ├── evaluate.py               Module 2b: Evaluation & reporting CLI
│   ├── predict.py                Module 3: Recognition/inference CLI
│   ├── generate_demo_dataset.py  Synthetic data for pipeline smoke-tests
│   └── live_demo.py              Optional live webcam demo (bonus)
├── data/
│   ├── raw/<CLASS_LABEL>/*.jpg   Your own collected images go here
│   └── processed/landmarks_dataset.csv   Extracted feature dataset
├── models/                      Trained model, label encoder, metrics
├── docs/
│   ├── DESIGN.md                 Full design & documentation write-up
│   └── diagrams/                 Architecture / workflow / UML / schema diagrams
├── tests/
│   └── test_pipeline.py         Unit tests covering every module
└── samples/                      Example commands / test files go here
```

## 4. Prerequisites

- Python 3.9 – 3.11 (MediaPipe's classic API wheels target this range)
- pip
- A webcam is only needed for the optional live/webcam modes; everything
  else works with static image/video files.

## 5. Setup

```bash
# 1. Clone the repository
git clone https://github.com/manish42sharma/Computer-vision-vityarthi-project.git
cd Computer-vision-vityarthi-project

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **MediaPipe version note:** `requirements.txt` pins a MediaPipe release
> that ships the classic `mediapipe.solutions.hands` API used by this
> project. If your platform only has a newer MediaPipe available (which
> ships a different "Tasks" API instead), `src/feature_extraction.py`
> auto-detects this and falls back automatically — see
> **Troubleshooting** below for the one extra download step that mode
> needs.

## 6. Quick Start (no camera needed — verify everything works)

```bash
# Generate a small synthetic landmark dataset (5 classes, ~1 second)
python -m src.generate_demo_dataset --samples-per-class 150

# Train a classifier on it
python -m src.train_model --model-type random_forest

# Evaluate it
python -m src.evaluate
```

You should see a training summary and a classification report printed
to the terminal, and `models/sign_classifier.pkl` created. This
confirms your environment is set up correctly before you invest time
collecting real gesture data.

## 7. Building a Real Dataset

### Option A — from a folder of your own photos (fully headless)

Arrange photos of a hand making each sign into one folder per class:

```
data/raw/
├── A/  img001.jpg  img002.jpg  ...
├── B/  img001.jpg  ...
├── C/  ...
```

Then run:

```bash
python -m src.collect_data --source images --raw-dir data/raw
```

### Option B — from your webcam (one class at a time)

```bash
# Records 15 seconds of landmark samples labelled "A"
python -m src.collect_data --source webcam --label A --seconds 15
python -m src.collect_data --source webcam --label B --seconds 15
# ... repeat for every class you want to recognise
```

Add `--display` if you're at a machine with a screen and want a live
preview while recording; it's optional and off by default so the
command still works over SSH / in CI.

## 8. Training and Evaluating on Your Real Dataset

```bash
python -m src.train_model --model-type random_forest --cv-folds 5
python -m src.evaluate
```

Try `--model-type svm` or `--model-type mlp` to compare algorithms — all
three are implemented in `src/model.py`.

## 9. Recognising a Sign (the main deliverable)

```bash
# A single image
python -m src.predict --input samples/test_A.jpg

# Every image in a folder
python -m src.predict --input samples/ --mode folder

# A video file, saving per-frame predictions to a CSV
python -m src.predict --input samples/demo.mp4 --mode video --output-csv results.csv
```

Example output:

```
=== Recognition Results ===
samples/test_A.jpg                            -> A          (confidence: 0.94)

Most likely sign: 'A' (1/1 detected frames agree)

Processed 1 frame(s)/image(s) in 0.18s (5.6 fps).
```

`predict.py` never requires a display — `--display` on video mode is a
purely optional live overlay for demos on machines that have a screen.

## 10. Optional Live Webcam Demo

```bash
python -m src.live_demo --camera-index 0
```

Press `q` to quit. This is a bonus interactive mode; it is **not**
required for, and not used by, the headless CLI evaluation path above.

## 11. Running the Tests

```bash
pip install pytest   # already in requirements.txt
pytest tests/ -v
```

The suite covers: landmark-vector normalisation, dataset loading and
stratified splitting, model training/persistence/round-trip inference,
and error handling for missing files and invalid arguments.

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: mediapipe is required...` | `pip install -r requirements.txt` |
| `FileNotFoundError: ... hand_landmarker.task ...` | Your installed MediaPipe only has the newer "Tasks" API. Download the model from the URL printed in the error message and save it to `models/hand_landmarker.task`, **or** `pip install mediapipe==0.10.14` for the classic API instead. |
| `RuntimeError: Could not open camera index 0` | No webcam available/permitted. Use `--source images` (collection) or `--mode image/folder/video` (prediction) instead. |
| "No hand detected" on every frame | Improve lighting, keep the whole hand in frame, or lower `MIN_DETECTION_CONFIDENCE` in `src/config.py`. |
| Low accuracy after training | Collect more samples per class (aim for 100+), keep hand framing/background consistent, and re-run `evaluate.py` to check per-class precision/recall. |

## 13. Extending the Project

- Add more classes: just create more `data/raw/<LABEL>/` folders or
  record more webcam sessions, then retrain.
- Swap the classifier: add a case to `build_model()` in `src/model.py`.
- Two-hand or dynamic (motion) signs: extend `feature_extraction.py` to
  track a second hand and/or a temporal window of frames, and swap the
  classifier for a sequence model — see `docs/DESIGN.md` → "Future
  Enhancements" for the full rationale.

## 14. License / Academic Integrity

Submitted as original coursework for the Computer Vision course
"Build Your Own Project" evaluation. All code in this repository was
authored for this assignment.

## 15. Author
Manish (24BAI10899)
