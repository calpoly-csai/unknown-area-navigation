# Methods — Unknown Object Detection

## Overview

Standard object detection models operate on closed-vocabulary datasets. When presented with an object outside their training distribution — a niche tool, an unusual prop, or a novel obstacle — these models either misclassify the object or assign low confidence to their best guess. The goal of this subsystem is to detect when a model is uncertain and flag the object as *unknown*, rather than silently producing a wrong label.

Three approaches were implemented, ranging from a closed-vocabulary baseline to a dual-model consensus pipeline.

---

## Study Design

This is an **exploratory, systems-development study**. Approaches were implemented as live webcam prototypes and evaluated qualitatively by presenting niche or unusual objects to the camera and observing detection behavior. No standardized benchmark dataset was used in this phase.

---

## Approach 1 — Confidence-Thresholded YOLO-World

**Implementation:** `src/unknown_object_detection/ObjectRecognition.py`

**Model:** YOLOv8s-worldv2 (YOLO-World, small variant)

YOLO-World is an open-vocabulary object detector capable of generalizing beyond fixed class lists. The model was loaded via the Ultralytics library and run on a live webcam stream using OpenCV.

### Detection Logic

- Each frame was passed through the model with agnostic non-maximum suppression (`agnostic_nms=True`) and an IoU threshold of 0.6 to suppress overlapping detections.
- For each bounding box, the model's confidence score was extracted.
- A confidence threshold of **50%** was applied:
  - Detections **≥ 50%** confidence → labeled with the predicted class name (blue text).
  - Detections **< 50%** confidence → labeled **"UNKNOWN OBJECT"** (red text).
- Annotated frames were displayed in real time.

### Rationale

Confidence score acts as a proxy for the model's certainty. Objects outside the model's training distribution tend to produce lower confidence scores as the model attempts to match them to the nearest known class.

---

## Approach 2 — Dual-Model Consensus (NanoOWL + YOLO-World)

**Implementation:** `src/unknown_object_detection/nano_owl_vs_yolo_world.py`

**Models:**
- **NanoOWL** (`google/owlvit-base-patch32`) — a class-agnostic, open-vocabulary detector based on OWL-ViT, optimized for NVIDIA edge hardware
- **YOLO-World** (`yolov8s-worldv2.pt`) — an open-vocabulary detector queried against a fixed 12-class candidate list

### Core Idea

A class-agnostic model detects *whether* an object is present; a traditional model attempts to *classify* it. If the agnostic model finds an object but the traditional model cannot match it with sufficient confidence, the object is flagged as unknown.

### Pipeline

1. Each webcam frame was resized to a maximum width of 640 pixels to reduce inference latency.
2. **NanoOWL** was run every 3 frames using two prompts in sequence:
   - Primary prompt: `["objects"]` — detects any generic object presence.
   - Fallback prompt: `["handheld object"]` — used if the primary prompt yields no detections.
   - Detection threshold: 0.08 (permissive, to maximize recall).
3. **YOLO-World** was run on every frame against a fixed candidate class list:
   `person, cell phone, remote, book, bottle, cup, mouse, keyboard, pen, marker, tool, box`
   - Detections below a display threshold of 0.10 were discarded.
4. **IoU-based matching** was performed between each NanoOWL detection and all YOLO-World detections. The highest-overlap YOLO-World box was selected as the candidate match (IoU threshold: 0.20).
5. Each NanoOWL detection was classified into one of three outcomes:

| Outcome | Condition | Display Color |
|---|---|---|
| **Unknown** | No YOLO-World match, or IoU < 0.20 | Red |
| **Weak match** | YOLO-World match found, confidence < 0.35 | Yellow |
| **Matched** | YOLO-World match found, confidence ≥ 0.35 | Green |

6. Per-frame summary statistics (matched, weak match, unknown counts) were overlaid on the display.

### Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `NANO_OWL_THRESHOLD` | 0.08 | Permissive detection threshold for NanoOWL |
| `YOLO_WORLD_SHOW_THRESHOLD` | 0.10 | Minimum confidence to display a YOLO detection |
| `YOLO_WORLD_LOW_CONF_THRESHOLD` | 0.35 | Threshold separating weak from strong YOLO matches |
| `IOU_MATCH_THRESHOLD` | 0.20 | Minimum IoU to count a NanoOWL/YOLO box as the same object |
| `INFERENCE_MAX_WIDTH` | 640 px | Frame resize cap for inference speed |
| `NANO_OWL_FRAME_INTERVAL` | 3 | Run NanoOWL every N frames to reduce compute load |

---

## Approach 3 — Baseline Closed-Vocabulary YOLO

**Implementation:** `src/unknown_object_detection/YOLO.py`

**Model:** YOLOv11n (nano, closed-vocabulary)

A standard YOLO model was run on a live webcam feed with no unknown-object logic. Detections were annotated using the model's built-in class set. This baseline illustrates the core limitation motivating Approaches 1 and 2: the model produces labels for everything it sees, with no mechanism to express uncertainty or flag out-of-distribution objects.

---

## Study Materials

| Component | Details |
|---|---|
| Camera input | Live webcam feed (`cv2.VideoCapture(0)`) |
| Test objects | Niche/unusual items (e.g., USB adapters, marker caps, measuring tape, earbuds cases, hobby tools) presented against plain backgrounds |
| Hardware | Standard laptop/desktop CPU; NanoOWL configured for `device="cpu"` |
| Frameworks | Ultralytics, NanoOWL, OpenCV, PyTorch, Pillow |

---

## Limitations

- Evaluated qualitatively; no standardized benchmark dataset was used.
- NanoOWL was run on CPU, which is significantly slower than the intended NVIDIA edge hardware target.
- The YOLO-World candidate class list (12 classes) was manually curated and may not generalize to all environments.
- A planned third approach — training a custom class-agnostic model on a researcher-collected dataset (20–30 objects labeled under a single "Object" class via AnyLabeling/Roboflow) — was designed but not completed in this phase. The partial implementation is in `src/unknown_object_detection/agnostic_nano_owl.py`.
