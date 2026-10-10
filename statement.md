# Problem Statement

## Problem Statement

Communication between Deaf/hard-of-hearing sign-language users and people who
do not know sign language is often limited to an interpreter or written text.
There is a need for lightweight, low-cost software that can look at a hand
gesture — from a photo, a video clip, or a webcam — and tell the user, in
real time, which sign it most likely represents, without requiring
specialised hardware (gloves, depth sensors) or a GPU.

This project builds a **Sign Language Recognition System** for the Computer
Vision course: a command-line pipeline that captures hand images, extracts
geometric hand-landmark features using MediaPipe, trains a classical machine
learning classifier (Random Forest / SVM / MLP, selectable) on those
features, and recognises static signs (the fingerspelling alphabet, e.g.
A–E as a demonstrable subset, extensible to the full alphabet or a custom
gesture vocabulary) from new images, videos, or a live camera feed.

## Scope of the Project

**In scope**
- Static, single-hand gesture recognition (fingerspelling-style signs).
- A full offline pipeline: data collection → feature extraction → training
  → evaluation → inference, runnable entirely from the command line.
- Recognition from a single image, a folder of images, or a video file.
- An optional live-webcam demo for interactive presentation.
- Extensibility to any number of gesture classes the user chooses to
  collect data for (not hard-coded to a fixed alphabet).

**Out of scope** (documented explicitly to set correct expectations for
evaluation)
- Continuous / dynamic sign-language sentence recognition (temporal signs
  that involve motion trajectories, e.g. full ASL grammar) — this would
  require a sequence model (e.g. an LSTM/Transformer over landmark
  sequences) and a much larger, professionally annotated dataset; it is
  flagged under "Future Enhancements" in the project report.
- Two-handed signs and facial-expression-dependent signs.
- Deployment as a mobile app or web service (the deliverable is a CLI tool
  and reusable Python package, per the assignment's "must run from the
  command line" requirement).

## Target Users

- **Sign-language learners** who want quick, private feedback on whether
  their hand shape matches a target sign.
- **Developers/researchers** who want a clean, well-documented reference
  implementation of a landmark-based gesture classifier to extend.
- **Course evaluators** who need to run the project headlessly, from a
  terminal, with no GUI dependency, and verify its correctness against a
  test set.

## High-Level Features

1. **Data Collection & Feature Extraction** — capture hand images (from a
   folder of photos or a webcam) and convert each into a normalised
   63-dimensional landmark feature vector (21 hand landmarks × x, y, z).
2. **Model Training & Evaluation** — train a classifier on the extracted
   features with a train/test split and k-fold cross-validation, then
   produce a classification report and confusion matrix.
3. **Sign Recognition (Inference)** — given a new image, folder, or video,
   output the predicted sign label and a confidence score, entirely via the
   command line, with an optional CSV export and an optional live-preview
   window for interactive demos.
