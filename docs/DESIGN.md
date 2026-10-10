# Design & Documentation — Sign Language Recognition System

This document contains the full set of design artefacts required by the
project brief. See `statement.md` for the problem statement/scope, and
`docs/diagrams/` for all referenced figures.

## 1. Objectives

1. Recognise static hand-sign gestures from an image, video, or webcam
   feed using classical computer-vision + machine-learning techniques.
2. Keep the entire pipeline runnable from a terminal, on a CPU-only
   machine, with no mandatory GUI.
3. Make the system easy to extend to new gesture vocabularies without
   changing code (only new data + retraining).
4. Demonstrate sound software-engineering practice: modular code,
   logging, error handling, automated tests, and reproducible results.

## 2. Functional Requirements

| ID | Requirement | Module |
|----|-------------|--------|
| FR1 | The system shall extract 21 hand landmarks (x, y, z) from a BGR image/frame using MediaPipe. | `feature_extraction.py` |
| FR2 | The system shall normalise landmarks for translation and scale invariance before use as features. | `feature_extraction.py` |
| FR3 | The system shall build a labelled dataset from either a folder of images or a webcam capture session. | `collect_data.py` |
| FR4 | The system shall train a classifier (Random Forest, SVM, or MLP — user-selectable) on the labelled dataset. | `train_model.py`, `model.py` |
| FR5 | The system shall report train/test accuracy, cross-validation scores, per-class precision/recall/F1, and a confusion matrix. | `train_model.py`, `evaluate.py` |
| FR6 | The system shall persist the trained model and label encoder so inference does not require retraining. | `model.py` |
| FR7 | The system shall predict the sign shown in a single image, every image in a folder, or every sampled frame of a video, and report a confidence score. | `predict.py` |
| FR8 | The system shall optionally export predictions to a CSV file. | `predict.py` |
| FR9 | The system shall optionally display a live annotated webcam feed for interactive demonstration. | `live_demo.py` (bonus) |

This satisfies "three major functional modules" (Data Collection,
Training/Evaluation, Recognition) with a clear input → processing →
output structure in each (see §5, Workflow Diagram, and §7, Sequence
Diagram).

## 3. Non-Functional Requirements

| Category | Requirement | How it is met |
|---|---|---|
| **Performance** | Real-time-capable feature extraction (MediaPipe runs well over 15–30 FPS on CPU for a single hand). | Lightweight 63-D landmark vector instead of raw-pixel CNN inference; `FRAME_SKIP` config to trade accuracy vs. throughput on video. |
| **Reliability** | The system must not crash on bad/missing input, an absent camera, or an undetected hand. | `safe_run` decorator (utils.py) catches and logs errors with clean exit codes; every extraction call returns `(None, None)` rather than raising on a missed detection. |
| **Usability** | Every CLI script must be self-documenting and runnable with no prior context. | `argparse` `--help` on every script; sensible defaults; example commands in `README.md`. |
| **Scalability** | Adding new gesture classes must not require code changes. | Classes are discovered dynamically from `data/raw/<label>/` folder names or webcam `--label` argument; `LabelEncoder` adapts automatically. |
| **Maintainability** | Configuration and logic must not be duplicated across scripts. | All tunables centralised in `config.py`; model construction centralised in `model.py`; every script imports shared logic rather than re-implementing it. |
| **Error-handling strategy** | Predictable, actionable failures. | `utils.safe_run` maps `FileNotFoundError`/`ValueError`/unexpected exceptions to distinct exit codes (2/3/1) with a logged, human-readable message. |
| **Logging/monitoring** | Every run must leave an audit trail. | `utils.get_logger` writes timestamped logs to both stdout and `logs/sign_language_system.log`; `utils.timeit` logs the duration of every major operation. |
| **Resource efficiency** | Must run on a laptop CPU, no GPU. | Landmark-based features (63 floats) instead of image tensors; Random Forest/SVM/MLP instead of a deep CNN; training on hundreds of samples completes in well under a second (verified — see report §11, "Testing Approach"). |

## 4. System Architecture

See `docs/diagrams/architecture_diagram.png`.

The system is organised into four layers:

1. **Input Layer** — a webcam or static image/video files.
2. **Computer Vision Layer** — MediaPipe Hands locates 21 landmarks per
   detected hand; a normalisation step removes translation and scale
   dependence so the same sign looks numerically similar regardless of
   hand position/distance from the camera.
3. **Machine Learning Layer** — a scikit-learn classifier is trained on
   the normalised landmark vectors and persisted to disk; a companion
   evaluation module reports accuracy metrics.
4. **Application Layer (CLI)** — `predict.py` ties feature extraction and
   the trained model together to answer "what sign is this?" for any
   image, folder, or video, entirely from the terminal.

## 5. Process Flow / Workflow

See `docs/diagrams/workflow_diagram.png`. In short: **collect → extract
→ build dataset → train → evaluate → (loop back if accuracy is
insufficient) → predict → output**.

## 6. UML Diagrams

- **Use Case Diagram** — `docs/diagrams/use_case_diagram.png`. Two
  actors (Student/End User, Course Evaluator) interact with seven use
  cases; "Train classifier" includes "Evaluate model accuracy", and both
  recognition use cases extend to "Export predictions to CSV".
- **Class Diagram** — `docs/diagrams/class_diagram.png`. Shows the two
  stateful classes (`HandLandmarkExtractor`, `SignPredictor`) and the
  five functional modules that depend on them and on the shared
  `config` module.
- **Sequence Diagram** — `docs/diagrams/sequence_diagram.png`. Traces a
  single `predict.py --input photo.jpg` call end-to-end: model/encoder
  loading, landmark extraction, classification, and result output.

## 7. Data / Storage Design

This project intentionally uses **flat-file storage** (CSV + pickled
scikit-learn objects) rather than a relational database, because the
data is a single, simply-structured feature table with no relational
structure, concurrent-write requirements, or query needs beyond "load
the whole table" — a database would add operational complexity with no
benefit for a coursework-scale dataset (see §9 for the alternatives
considered). The schema is documented as a record diagram in
`docs/diagrams/data_schema_diagram.png` and summarised here:

**`data/processed/landmarks_dataset.csv`** — one row per collected hand
pose:

| Column | Type | Description |
|---|---|---|
| `x0, y0, z0` … `x20, y20, z20` | float32 | 21 hand landmarks × 3 coordinates, wrist-relative and scale-normalised (63 columns total) |
| `label` | string | The gesture class this sample belongs to |

**`models/sign_classifier.pkl`** — the trained scikit-learn estimator.
**`models/label_encoder.pkl`** — the `LabelEncoder` mapping label strings
to the integer indices the model was trained on.
**`models/metrics.json`** — training run metadata (accuracy, sample
counts, timings) for auditability.

## 8. Dataset Description (Computation-Heavy / ML Component)

- **Real deployment dataset**: hand-landmark vectors collected by the
  user via `collect_data.py`, either from their own photographs (one
  folder per class under `data/raw/`) or from a short webcam session per
  class. Each sample is a 63-dimensional vector, so even a few dozen
  samples per class is enough for the classical classifiers used here —
  unlike raw-pixel CNN approaches, which typically need thousands of
  images per class.
- **Demo/smoke-test dataset**: `generate_demo_dataset.py` produces a
  synthetic dataset (Gaussian clusters in the same 63-D feature space)
  purely so the training/evaluation/inference code paths can be
  verified end-to-end without a camera. It is clearly not a substitute
  for real gesture data and is documented as such in its own module
  docstring and in the Testing section of the report.

## 9. Model Selection Rationale

| Option considered | Verdict | Reasoning |
|---|---|---|
| CNN on raw pixels (e.g. a small ResNet) | Rejected for this scope | Needs thousands of labelled images per class, a GPU for practical training time, and is far more sensitive to background/lighting — poor fit for a student-collected dataset and a CLI-only, CPU-only deliverable. |
| MediaPipe landmarks + classical ML (chosen) | **Selected** | 63-D structured features are already viewpoint-normalised by MediaPipe's own detector; classical models train in milliseconds on a CPU and need only tens-to-hundreds of samples per class. |
| Landmarks + Random Forest | **Default** | Robust to noisy/irrelevant landmark jitter, needs no feature scaling, trains almost instantly, and gives interpretable per-class metrics. |
| Landmarks + SVM (RBF) | Offered as an alternative (`--model-type svm`) | Often slightly better decision boundaries on small, well-separated datasets, at the cost of needing careful `C`/`gamma` tuning for best results. |
| Landmarks + MLP | Offered as an alternative (`--model-type mlp`) | Useful if the gesture vocabulary grows large and class boundaries become more non-linear; more prone to overfitting on very small datasets. |

`model.py` implements all three behind one factory function, so
switching is a single CLI flag — the comparison itself is left as an
easy experiment for the user rather than baked into one "right answer".

## 10. Evaluation Methodology

1. **Stratified train/test split** (80/20 by default, `config.TEST_SIZE`)
   keeps class proportions balanced in both halves, which matters on a
   small, manually collected dataset.
2. **k-fold cross-validation** (`config.CROSS_VAL_FOLDS`, default 5) on
   the training split gives a variance-aware estimate of generalisation
   accuracy, not just a single lucky/unlucky split.
3. **Held-out test accuracy** plus a full **`sklearn.metrics.classification_report`**
   (precision/recall/F1 per class) and **confusion matrix** are produced
   by `evaluate.py`, so weaknesses on specific, easily-confused signs
   (e.g. visually similar hand shapes) are visible rather than hidden
   behind one aggregate accuracy number.
4. Metrics are persisted to `models/metrics.json` and
   `models/evaluation_report.json` for reproducibility and for direct
   inclusion in the project report.

## 11. Design Decisions & Trade-offs (Summary)

- **Landmarks over raw pixels** — see §9. Chosen for data-efficiency and
  CPU-only feasibility.
- **Flat-file storage over a database** — see §7. Chosen because the
  data is a single simple table with no relational or concurrency needs.
- **CLI-first, GUI-optional** — every required capability
  (`collect_data.py --source images`, `train_model.py`, `evaluate.py`,
  `predict.py`) runs headlessly by default; any `--display`/webcam
  window is strictly additive, satisfying the "must be executable via
  command line" / "penalised for GUI-only setup" submission rule.
- **Dual MediaPipe API support** — MediaPipe's Python API changed
  across releases (a classic `solutions.hands` API vs. a newer `Tasks`
  API requiring a downloaded model file). `feature_extraction.py`
  detects which is installed and adapts, rather than hard-coding one and
  breaking on the other — see "Challenges Faced" in the project report
  for the full story.

## 12. Future Enhancements

- Extend from static fingerspelling signs to dynamic, motion-based signs
  using a sequence model (LSTM/Transformer) over a sliding window of
  landmark frames.
- Support two-handed signs by extracting and concatenating landmarks for
  both detected hands.
- Add a confidence-calibrated "reject" class more rigorously (e.g.
  Platt scaling) instead of a fixed probability threshold.
- Package the trained model behind a small REST API for integration
  into other applications, while keeping this CLI as the reference
  implementation.
