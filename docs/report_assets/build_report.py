"""
Builds the project report PDF (docs/report_assets -> Project_Report.pdf)
following the 15-section structure required by the submission guidelines.
"""

import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    ListFlowable, ListItem, HRFlowable, KeepTogether, Preformatted
)
import textwrap

DIAG = "docs/diagrams"
ASSETS = "docs/report_assets"

with open("models/metrics.json") as f:
    metrics = json.load(f)
with open("models/evaluation_report.json") as f:
    eval_report = json.load(f)
with open(f"{ASSETS}/terminal_transcript.txt") as f:
    transcript = f.read()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCover", fontSize=26, leading=32, alignment=TA_CENTER,
                           spaceAfter=14, textColor=colors.HexColor("#1F3864"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="SubtitleCover", fontSize=14, leading=20, alignment=TA_CENTER,
                           textColor=colors.HexColor("#44546A"), spaceAfter=6))
styles.add(ParagraphStyle(name="H1", fontSize=17, leading=21, spaceBefore=14, spaceAfter=8,
                           textColor=colors.HexColor("#1F3864"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6,
                           textColor=colors.HexColor("#2E5395"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="BodyText2", fontSize=10.3, leading=15, spaceAfter=6, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="TermCode", fontSize=8.3, leading=10.6, fontName="Courier",
                           backColor=colors.HexColor("#F2F2F2"), borderPadding=8,
                           leftIndent=2, rightIndent=2))
styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=12, alignment=TA_CENTER,
                           textColor=colors.HexColor("#595959"), spaceAfter=14, fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle(name="CellText", fontSize=8.3, leading=10.6, fontName="Helvetica"))
styles.add(ParagraphStyle(name="CellTextBold", fontSize=8.3, leading=10.6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="CellCode", fontSize=8.0, leading=10.2, fontName="Courier"))
styles.add(ParagraphStyle(name="CellHeader", fontSize=8.6, leading=11, fontName="Helvetica-Bold", textColor=colors.white))


def cell(text, style="CellText"):
    """Wrap plain text in a Paragraph so Table cells actually wrap
    within their column width instead of overflowing into the next
    column (the default behaviour for raw strings can mis-measure)."""
    return Paragraph(text, styles[style])

story = []

# ---------------------------------------------------------------------
# 1. Cover Page
# ---------------------------------------------------------------------
story.append(Spacer(1, 4 * cm))
story.append(Paragraph("Sign Language Recognition System", styles["TitleCover"]))
story.append(Paragraph("A Computer-Vision &amp; Machine-Learning Pipeline for Static Sign Recognition",
                        styles["SubtitleCover"]))
story.append(Spacer(1, 1.2 * cm))
story.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#8FAADC"), hAlign="CENTER"))
story.append(Spacer(1, 1.2 * cm))

cover_table_data = [
    [cell("Course", "CellTextBold"), cell("Computer Vision — Flipped Course Evaluation")],
    [cell("Assignment", "CellTextBold"), cell("VITyarthi — Build Your Own Project")],
    [cell("Project Title", "CellTextBold"), cell("Sign Language Recognition System")],
    [cell("Submission Type", "CellTextBold"), cell("GitHub Repository + Project Report (PDF)")],
]
t = Table(cover_table_data, colWidths=[4.2 * cm, 9.8 * cm])
t.setStyle(TableStyle([
    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1F3864")),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
]))
story.append(t)
story.append(PageBreak())

# ---------------------------------------------------------------------
# 2. Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("2. Introduction", styles["H1"]))
story.append(Paragraph(
    "Sign language is a primary means of communication for many Deaf and hard-of-hearing "
    "individuals, yet most software and devices cannot interpret it. This project applies "
    "computer-vision techniques taught in this course to build a practical, lightweight "
    "system that recognises static hand signs from an image, a video file, or a live webcam "
    "feed, entirely from the command line and without specialised hardware.", styles["BodyText2"]))
story.append(Paragraph(
    "The system uses <b>MediaPipe Hands</b> to locate 21 3-D landmarks on a detected hand in "
    "each frame, converts these into a normalised 63-dimensional feature vector, and trains a "
    "classical machine-learning classifier (Random Forest by default, with SVM and MLP also "
    "available) to map that feature vector to a gesture label. This landmark-based approach "
    "was deliberately chosen over a raw-pixel convolutional neural network, because it needs "
    "orders of magnitude less training data and runs comfortably on a CPU-only laptop — both "
    "important constraints for a student project built and evaluated without dedicated "
    "GPU infrastructure.", styles["BodyText2"]))
story.append(Paragraph(
    "The remainder of this report documents the problem statement and scope, the functional "
    "and non-functional requirements, the system's architecture and design (including UML "
    "diagrams), the implementation, the testing approach and results, the challenges "
    "encountered, and possible future enhancements.", styles["BodyText2"]))

# ---------------------------------------------------------------------
# 3. Problem Statement
# ---------------------------------------------------------------------
story.append(Paragraph("3. Problem Statement", styles["H1"]))
story.append(Paragraph(
    "Communication between sign-language users and people unfamiliar with sign language "
    "typically depends on a human interpreter or written text. There is a need for accessible, "
    "low-cost software that can look at a hand gesture and identify which sign it represents, "
    "without specialised hardware such as data gloves or depth sensors.", styles["BodyText2"]))
story.append(Paragraph(
    "This project builds a Sign Language Recognition System that captures hand images, "
    "extracts geometric hand-landmark features, trains a classifier on those features, and "
    "recognises static signs (a fingerspelling-style alphabet, demonstrated on a 5-class "
    "subset and freely extensible to more classes) from new images, videos, or a live camera "
    "feed — as a fully command-line-driven pipeline.", styles["BodyText2"]))
story.append(Paragraph("<b>In scope:</b> static single-hand gesture recognition; an offline, "
                        "CLI-driven pipeline covering data collection, training, evaluation and "
                        "inference; recognition from images, folders of images, or video files; "
                        "an optional interactive webcam demo; and extensibility to any number of "
                        "gesture classes.", styles["BodyText2"]))
story.append(Paragraph("<b>Out of scope:</b> continuous/dynamic sentence-level sign-language "
                        "recognition requiring temporal sequence models and large annotated "
                        "corpora; two-handed and facial-expression-dependent signs; and "
                        "deployment as a mobile/web application (the deliverable is a CLI tool "
                        "and reusable Python package). These are revisited under "
                        "\"Future Enhancements\".", styles["BodyText2"]))

# ---------------------------------------------------------------------
# 4. Functional Requirements
# ---------------------------------------------------------------------
story.append(Paragraph("4. Functional Requirements", styles["H1"]))
fr_data = [["ID", "Requirement", "Module"]]
fr_rows = [
    ("FR1", "Extract 21 hand landmarks (x, y, z) from a BGR image/frame using MediaPipe.", "feature_extraction.py"),
    ("FR2", "Normalise landmarks for translation and scale invariance before use as features.", "feature_extraction.py"),
    ("FR3", "Build a labelled dataset from a folder of images or a webcam capture session.", "collect_data.py"),
    ("FR4", "Train a classifier (Random Forest / SVM / MLP, user-selectable) on the dataset.", "train_model.py, model.py"),
    ("FR5", "Report train/test accuracy, cross-validation scores, per-class metrics and a confusion matrix.", "train_model.py, evaluate.py"),
    ("FR6", "Persist the trained model and label encoder so inference does not require retraining.", "model.py"),
    ("FR7", "Predict the sign in an image, every image in a folder, or sampled video frames, with a confidence score.", "predict.py"),
    ("FR8", "Optionally export predictions to a CSV file.", "predict.py"),
    ("FR9", "Optionally display a live annotated webcam feed (bonus, not required for CLI grading).", "live_demo.py"),
]
for fid, req, mod in fr_rows:
    fr_data.append([cell(fid, "CellTextBold"), cell(req), cell(mod, "CellCode")])
fr_table = Table(fr_data, colWidths=[1.4 * cm, 9.4 * cm, 3.2 * cm], repeatRows=1)
fr_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 8.6),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FC")]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
story.append(fr_table)
story.append(Paragraph(
    "Together, FR1–FR3 form the <b>Data Collection &amp; Feature Extraction</b> module, "
    "FR4–FR6 the <b>Training &amp; Evaluation</b> module, and FR7–FR9 the "
    "<b>Recognition (Inference)</b> module — the three major functional modules required "
    "by the assignment brief.", styles["BodyText2"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# 5. Non-Functional Requirements
# ---------------------------------------------------------------------
story.append(Paragraph("5. Non-Functional Requirements", styles["H1"]))
nfr_data = [["Category", "Requirement", "How it is met"]]
nfr_rows = [
    ("Performance", "Real-time-capable feature extraction.",
     "63-D landmark vector instead of raw-pixel CNN inference; FRAME_SKIP config trades accuracy vs. throughput on video."),
    ("Reliability", "Must not crash on bad input, an absent camera, or a missed hand detection.",
     "safe_run decorator catches and logs errors with clean exit codes; extraction returns (None, None) rather than raising."),
    ("Usability", "Every CLI script must be self-documenting.",
     "argparse --help on every script; sensible defaults; documented example commands in README.md."),
    ("Scalability", "Adding gesture classes must not require code changes.",
     "Classes are discovered dynamically from folder names / --label; LabelEncoder adapts automatically."),
    ("Maintainability", "No duplicated configuration or logic across scripts.",
     "All tunables centralised in config.py; model construction centralised in model.py."),
    ("Error handling", "Predictable, actionable failures.",
     "safe_run maps exception types to distinct exit codes (2/3/1) with a logged, human-readable message."),
    ("Logging / monitoring", "Every run must leave an audit trail.",
     "get_logger writes timestamped logs to stdout and logs/sign_language_system.log; timeit logs operation durations."),
    ("Resource efficiency", "Must run on a CPU-only laptop.",
     "Landmark features (63 floats) instead of image tensors; classical ML instead of a deep CNN."),
]
for row in nfr_rows:
    cat, req, how = row
    nfr_data.append([cell(cat, "CellTextBold"), cell(req), cell(how)])
nfr_table = Table(nfr_data, colWidths=[2.6 * cm, 5.0 * cm, 6.4 * cm], repeatRows=1)
nfr_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 8.6),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FC")]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
story.append(nfr_table)
story.append(PageBreak())

# ---------------------------------------------------------------------
# 6. System Architecture
# ---------------------------------------------------------------------
story.append(Paragraph("6. System Architecture", styles["H1"]))
story.append(Paragraph(
    "The system is organised into four layers: an <b>Input Layer</b> (webcam or static "
    "image/video files), a <b>Computer Vision Layer</b> (MediaPipe hand-landmark detection "
    "and feature normalisation), a <b>Machine Learning Layer</b> (model training, persistence "
    "and evaluation), and an <b>Application Layer</b> exposing everything as command-line "
    "tools. Each layer communicates through a clear, serialisable interface — a normalised "
    "feature vector between the CV and ML layers, and a persisted model file between training "
    "and inference — so any layer can be modified or swapped independently.", styles["BodyText2"]))
story.append(Image(f"{DIAG}/architecture_diagram.png", width=17 * cm, height=17 * cm * 0.19))
story.append(Paragraph("Figure 1: System Architecture Diagram.", styles["Caption"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# 7. Design Diagrams
# ---------------------------------------------------------------------
story.append(Paragraph("7. Design Diagrams", styles["H1"]))

story.append(Paragraph("7.1 Use Case Diagram", styles["H2"]))
story.append(Paragraph(
    "Two actors interact with the system: the <b>Student/End User</b>, who collects data, "
    "trains models and recognises signs; and the <b>Course Evaluator</b>, who independently "
    "verifies model accuracy and recognition correctness. \"Train classifier\" includes "
    "\"Evaluate model accuracy\"; both recognition use cases extend to \"Export predictions "
    "to CSV\".", styles["BodyText2"]))
story.append(Image(f"{DIAG}/use_case_diagram.png", width=15.5 * cm, height=15.5*cm*(470/878)))
story.append(Paragraph("Figure 2: Use Case Diagram.", styles["Caption"]))

story.append(Paragraph("7.2 Process Flow / Workflow Diagram", styles["H2"]))
story.append(Image(f"{DIAG}/workflow_diagram.png", width=8.5 * cm, height=8.5*cm*(973/509)))
story.append(Paragraph("Figure 3: Process Flow / Workflow Diagram.", styles["Caption"]))
story.append(PageBreak())

story.append(Paragraph("7.3 Sequence Diagram", styles["H2"]))
story.append(Paragraph(
    "The sequence diagram below traces a single <code>predict.py --input photo.jpg</code> "
    "call: the predictor loads the persisted model and label encoder, extracts and normalises "
    "hand landmarks from the frame, obtains class probabilities from the model, and returns "
    "the predicted label and confidence to the CLI for output.", styles["BodyText2"]))
story.append(Image(f"{DIAG}/sequence_diagram.png", width=17 * cm, height=17*cm*(0.62)))
story.append(Paragraph("Figure 4: Sequence Diagram — Sign Recognition from a Single Image.", styles["Caption"]))
story.append(PageBreak())

story.append(Paragraph("7.4 Class / Component Diagram", styles["H2"]))
story.append(Image(f"{DIAG}/class_diagram.png", width=17 * cm, height=17*cm*(497/1444)))
story.append(Paragraph("Figure 5: Class / Component Diagram.", styles["Caption"]))

story.append(Paragraph("7.5 Data / Storage Schema", styles["H2"]))
story.append(Paragraph(
    "No relational database is used (see \"Design Decisions &amp; Rationale\" for the "
    "reasoning); data instead persists as a flat feature-table CSV plus pickled model "
    "artefacts. The schema below plays the role an ER diagram would for a database-backed "
    "project.", styles["BodyText2"]))
story.append(Image(f"{DIAG}/data_schema_diagram.png", width=17 * cm, height=17*cm*(154/1470)))
story.append(Paragraph("Figure 6: Dataset / Storage Schema.", styles["Caption"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# 8. Design Decisions & Rationale
# ---------------------------------------------------------------------
story.append(Paragraph("8. Design Decisions &amp; Rationale", styles["H1"]))

decisions = [
    ("Landmarks over raw pixels", "A CNN on raw pixels would need thousands of labelled "
     "images per class and a GPU for practical training time, and is far more sensitive to "
     "background/lighting changes. MediaPipe's 21-point hand landmarks are already "
     "viewpoint-normalised by the detector itself, so a classical model trained on just tens "
     "to hundreds of samples per class performs well — a much better fit for a "
     "student-collected dataset and a CPU-only deliverable."),
    ("Random Forest as the default classifier", "Random Forest is robust to noisy landmark "
     "jitter, needs no feature scaling, trains in milliseconds, and yields readily "
     "interpretable per-class metrics. SVM and MLP are implemented behind the same interface "
     "(model.py) so they remain a one-flag experiment (--model-type svm / mlp) rather than "
     "being ruled out."),
    ("Flat-file storage instead of a database", "The data is a single, simply-structured "
     "feature table (fixed 63 numeric columns + a label) with no relational structure, no "
     "concurrent-write requirements, and no query pattern beyond \"load the whole table\". "
     "A relational database would add operational complexity (schema migrations, a server "
     "process, connection handling) without any corresponding benefit at this scale."),
    ("CLI-first, GUI strictly optional", "Every required capability — data collection from "
     "images, training, evaluation, and recognition from an image/folder/video — runs "
     "headlessly by default. Any preview window (--display) is additive only, so the "
     "submission satisfies the \"must be executable via command line\" / \"no GUI-only setup\" "
     "rule even on a machine with no display attached."),
    ("Dual MediaPipe API support", "MediaPipe's Python API changed across releases: older "
     "versions expose a classic mediapipe.solutions.hands API, while some newer releases ship "
     "only a Tasks API that needs an explicitly downloaded model file. feature_extraction.py "
     "detects which API is available at import time and adapts automatically, rather than "
     "hard-coding one and silently breaking on the other — see \"Challenges Faced\" for how "
     "this was discovered."),
]
for title, body in decisions:
    story.append(Paragraph(f"<b>{title}.</b> {body}", styles["BodyText2"]))

# ---------------------------------------------------------------------
# 9. Implementation Details
# ---------------------------------------------------------------------
story.append(Paragraph("9. Implementation Details", styles["H1"]))
story.append(Paragraph(
    "The project is implemented in Python 3 as an installable-style package (src/) with "
    "eleven modules plus a test suite, summarised below.", styles["BodyText2"]))

impl_data = [["File", "Responsibility"]]
impl_rows = [
    ("config.py", "Central configuration: paths, MediaPipe thresholds, model hyperparameters."),
    ("utils.py", "Shared logging setup, timing decorator, and the safe_run error-handling decorator."),
    ("feature_extraction.py", "HandLandmarkExtractor: dual-backend MediaPipe wrapper producing normalised 63-D vectors."),
    ("collect_data.py", "CLI: builds the labelled dataset from images or webcam capture."),
    ("dataset.py", "Loads the dataset CSV, encodes labels, produces stratified train/test splits."),
    ("model.py", "Model factory (Random Forest / SVM / MLP) and persistence helpers."),
    ("train_model.py", "CLI: trains a classifier, cross-validates it, and saves it with metrics."),
    ("evaluate.py", "CLI: re-evaluates a saved model, producing a classification report and confusion matrix."),
    ("predict.py", "CLI: the SignPredictor class and image/folder/video recognition entry point."),
    ("generate_demo_dataset.py", "CLI: synthetic dataset generator for pipeline smoke-testing."),
    ("live_demo.py", "Optional bonus CLI: live annotated webcam loop."),
]
for row in impl_rows:
    fname, resp = row
    impl_data.append([cell(fname, "CellCode"), cell(resp)])
impl_table = Table(impl_data, colWidths=[4.6 * cm, 9.4 * cm], repeatRows=1)
impl_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 8.6),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FC")]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
story.append(impl_table)
story.append(Paragraph(
    "Each landmark vector is normalised by (a) subtracting the wrist landmark's coordinates "
    "from every point, making the representation translation-invariant, and (b) dividing by "
    "the maximum landmark distance from the wrist, making it scale-invariant — so the same "
    "hand shape produces a numerically similar feature vector regardless of where the hand "
    "sits in frame or how close it is to the camera.", styles["BodyText2"]))

# ---------------------------------------------------------------------
# 10. Screenshots / Results
# ---------------------------------------------------------------------
story.append(Paragraph("10. Screenshots / Results", styles["H1"]))
story.append(Paragraph(
    "Because the CLI is text-based by design, \"screenshots\" here are terminal transcripts "
    "and generated result plots rather than window captures. The run below uses the bundled "
    "synthetic demo dataset (generate_demo_dataset.py) purely to exercise the full pipeline "
    "end-to-end; results on a real, user-collected gesture dataset will naturally vary and are "
    "expected to be lower than this idealised, well-separated synthetic case (see \"Testing "
    "Approach\" and \"Challenges Faced\").", styles["BodyText2"]))

wrapped_lines = []
for line in transcript.strip().split("\n"):
    if len(line) <= 98:
        wrapped_lines.append(line)
    else:
        wrapped_lines.extend(textwrap.wrap(line, width=98, subsequent_indent="    ") or [""])
transcript_wrapped = "\n".join(wrapped_lines)
story.append(Preformatted(transcript_wrapped, styles["TermCode"]))

story.append(Spacer(1, 0.4 * cm))
story.append(Image(f"{DIAG}/confusion_matrix.png", width=9 * cm, height=9*cm*(750/900)))
story.append(Paragraph("Figure 7: Confusion Matrix on the held-out synthetic test split.", styles["Caption"]))
story.append(Image(f"{DIAG}/per_class_metrics.png", width=10.5 * cm, height=10.5*cm*(400/650)))
story.append(Paragraph("Figure 8: Per-class Precision / Recall / F1-score.", styles["Caption"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# 11. Testing Approach
# ---------------------------------------------------------------------
story.append(Paragraph("11. Testing Approach", styles["H1"]))
story.append(Paragraph(
    "Testing combined automated unit tests with manual, end-to-end CLI verification:", styles["BodyText2"]))
story.append(ListFlowable([
    ListItem(Paragraph("<b>Unit tests</b> (tests/test_pipeline.py, run with pytest) cover: "
                        "landmark-vector normalisation and translation invariance against "
                        "synthetic fake landmark objects; correct handling of blank/empty "
                        "frames; dataset loading and stratified train/test splitting; model "
                        "construction for all three supported algorithms and rejection of an "
                        "unknown algorithm name; full train → save → load → predict round-trip "
                        "equality; and that a missing dataset file raises a clear "
                        "FileNotFoundError.", styles["BodyText2"])),
    ListItem(Paragraph("<b>End-to-end CLI verification</b>: every script's --help output was "
                        "checked for a clear, complete argument surface; the full "
                        "generate → train → evaluate pipeline was run and its output captured "
                        "(§10); and predict.py's error-handling path was exercised against a "
                        "missing input file, confirming a clean, logged failure "
                        "with exit code 2 rather than an unhandled stack trace.", styles["BodyText2"])),
    ListItem(Paragraph("<b>Cross-validation</b> (5-fold, on the training split) was used "
                        "alongside a single held-out test split so the reported accuracy is "
                        "not a single lucky/unlucky split.", styles["BodyText2"])),
], bulletType="bullet"))
story.append(Paragraph(
    "On a real, hand-collected dataset, the same test suite and evaluate.py workflow apply "
    "directly — only the data source changes (collect_data.py against real photos/webcam "
    "footage rather than generate_demo_dataset.py's synthetic clusters).", styles["BodyText2"]))

# ---------------------------------------------------------------------
# 12. Challenges Faced
# ---------------------------------------------------------------------
story.append(Paragraph("12. Challenges Faced", styles["H1"]))
story.append(Paragraph(
    "<b>MediaPipe API divergence.</b> During development it became clear that different "
    "MediaPipe package releases expose two different Python APIs for hand tracking: an older "
    "\"solutions\" API (mediapipe.solutions.hands, used by the majority of published tutorials "
    "and courses) and a newer \"Tasks\" API that requires an explicitly downloaded .task model "
    "file with no Python-side auto-download. Code written against only one API would silently "
    "fail to import, or fail at runtime, on a machine with the other version installed. This "
    "was resolved by making feature_extraction.py detect which API is present at import time "
    "and route all landmark-extraction calls through the matching backend, with a clear, "
    "actionable error message (including the exact model download URL) if the Tasks API is "
    "present but no model file has been supplied yet.", styles["BodyText2"]))
story.append(Paragraph(
    "<b>Feature normalisation.</b> Early experimentation with raw (un-normalised) landmark "
    "coordinates showed the classifier partly learning where in the frame the hand happened "
    "to be, rather than its shape. Subtracting the wrist coordinate and dividing by the "
    "maximum landmark distance from it removed this confound and made the features properly "
    "shape-dependent rather than position-dependent.", styles["BodyText2"]))
story.append(Paragraph(
    "<b>Balancing GUI convenience against the \"CLI-only\" submission rule.</b> A live webcam "
    "preview is genuinely useful for demonstrating the project, but the assignment explicitly "
    "penalises GUI-dependent setups. The design resolves this by making every window "
    "(--display, live_demo.py) strictly optional and off by default, so the graded path never "
    "requires a display.", styles["BodyText2"]))

# ---------------------------------------------------------------------
# 13. Learnings & Key Takeaways
# ---------------------------------------------------------------------
story.append(Paragraph("13. Learnings &amp; Key Takeaways", styles["H1"]))
story.append(ListFlowable([
    ListItem(Paragraph("A well-chosen, structured feature representation (hand landmarks) can "
                        "make a classical, data-efficient ML model competitive with a "
                        "heavier deep-learning approach for a constrained problem — an "
                        "important lesson in matching model complexity to data availability.", styles["BodyText2"])),
    ListItem(Paragraph("Designing for headless/CLI execution from the start (rather than "
                        "retrofitting it onto a GUI-first prototype) is much simpler, and "
                        "forces cleaner separation between core logic and any optional "
                        "presentation layer.", styles["BodyText2"])),
    ListItem(Paragraph("Third-party library APIs are not always stable across versions; "
                        "defensive, auto-detecting integration code (as done for MediaPipe "
                        "here) is worth the extra effort for anything meant to run reliably "
                        "on other people's machines.", styles["BodyText2"])),
    ListItem(Paragraph("Centralising configuration and using consistent logging/error-handling "
                        "decorators across every CLI script substantially reduced repeated "
                        "code and made behaviour predictable across the whole pipeline.", styles["BodyText2"])),
], bulletType="bullet"))

# ---------------------------------------------------------------------
# 14. Future Enhancements
# ---------------------------------------------------------------------
story.append(Paragraph("14. Future Enhancements", styles["H1"]))
story.append(ListFlowable([
    ListItem(Paragraph("Extend from static fingerspelling signs to dynamic, motion-based signs "
                        "using a sequence model (e.g. LSTM/Transformer) over a sliding window "
                        "of landmark frames.", styles["BodyText2"])),
    ListItem(Paragraph("Support two-handed signs by extracting and concatenating landmarks for "
                        "both detected hands.", styles["BodyText2"])),
    ListItem(Paragraph("Replace the fixed confidence threshold with a properly calibrated "
                        "reject option (e.g. Platt scaling) for out-of-vocabulary gestures.", styles["BodyText2"])),
    ListItem(Paragraph("Expose the trained model behind a small REST API for integration into "
                        "other applications, while keeping this CLI as the reference "
                        "implementation.", styles["BodyText2"])),
], bulletType="bullet"))

# ---------------------------------------------------------------------
# 15. References
# ---------------------------------------------------------------------
story.append(Paragraph("15. References", styles["H1"]))
refs = [
    "Google. MediaPipe Hands / Hand Landmarker documentation. https://developers.google.com/mediapipe/solutions/vision/hand_landmarker",
    "scikit-learn developers. scikit-learn: Machine Learning in Python. https://scikit-learn.org/",
    "Bradski, G. The OpenCV Library. Dr. Dobb's Journal of Software Tools, 2000.",
    "Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2011.",
    "Zhang, F. et al. MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214, 2020.",
]
story.append(ListFlowable(
    [ListItem(Paragraph(r, styles["BodyText2"])) for r in refs],
    bulletType="1"
))

doc = SimpleDocTemplate(
    "docs/report_assets/Project_Report.pdf",
    pagesize=A4,
    leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    title="Sign Language Recognition System - Project Report",
    author="VITyarthi Submission",
)
doc.build(story)
print("Report built.")
