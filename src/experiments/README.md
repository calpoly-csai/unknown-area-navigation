# Experiments

## Standard Pipeline

All experiments in this folder follow the same abstract pipeline. What varies between experiments is the **unknown object detection method** (Stage 2) and the **multimodal model** (Stage 4).

```
┌─────────────────────────────────────────────────────┐
│  Stage 1 — Capture                                  │
│                                                     │
│  Open webcam stream via OpenCV                      │
│  Stages 2–5 run continuously in a loop              │
│  for every frame until the user exits               │
│  (e.g. presses ESC or q)                            │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Unknown Object Detection                 │
│                                                     │
│  Run detection method on the stream                 │
│  Classify objects as                                │
│    - Known   → no further action                    │
│    - Unknown → flag frame, proceed to Stage 3       │
└───────────────────────┬─────────────────────────────┘
                        │
              Unknown detected?
               │               │
              No               Yes
               │               │
               │               ▼
               │  ┌────────────────────────────────────┐
               │  │  Stage 3 — Persistence Gate        │
               │  │                                    │
               │  │  Detection models sometimes flag   │
               │  │  an unknown for less than a second │
               │  │  before dropping it;  this is      |
               │  |  likely a glitch and not an        |
               │  │  actual detection.                 |
               │  │  The persistence gate filters      │
               │  │  these out by requiring the        │
               │  │  unknown to appear consistently    │
               │  │  before it is treated as real.     |             
               │  │                                    |
               │  │                                    │
               │  │                                    |
               │  │  No  → continue loop               │
               │  │  Yes → capture frame snapshot      │
               │  └──────────────┬─────────────────────┘
               │                 │
               │                 ▼
               │  ┌────────────────────────────────────┐
               │  │  Stage 4 — Multimodal Query        │
               │  │                                    │
               │  │  Send snapshot to VLM              │
               │  │  Ask: is this traversable?         │
               │  │                                    │
               │  │  Receive: traversable (bool)       │
               │  │           justification (text)     │
               │  └──────────────┬─────────────────────┘
               │                 │
               └────────┬────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5 — Display                                  │
│                                                     │
│  Annotate frame with detection results              │
│  Show traversability verdict + justification        │
│  Render annotated frame to screen                   │
└───────────────────────┬─────────────────────────────┘
                        │
                  Exit key pressed?
                   │           │
                  No          Yes
                   │           │
                   └─── loop   └──► Release camera, exit
```



---

## Experiment Protocol

Each experiment is evaluated by presenting physical objects to the camera one at a time. The process is:

- Select **5–10 known objects** and **5–10 unknown objects** (see categories below)
- Hold each object in front of the camera for approximately **5 seconds**
- Record whether the system correctly classified it as known or unknown
- Repeat for all objects, then compute metrics

---

## Object Categories

**Known** — objects the detector should recognise and NOT flag as unknown:
- Headphones
- Water bottle
- Coffee mug
- Keyboard
- Mouse
- Book
- Phone
- Pen
- Backpack
- Person

**Unknown** — objects the detector should flag as unknown:
- Unusual hobby tool (e.g. soldering iron, rotary tool)
- Niche lab equipment (e.g. caliper, multimeter)
- Obscure kitchen gadget (e.g. avocado slicer, egg separator)
- Costume prop or novelty item
- Small electronic component (e.g. Arduino, sensor module)
- Unusual cable adapter or connector
- Object wrapped in tape or foil (disguised known)
- Broken/deformed version of a known object (e.g. crushed can, snapped pen)

---

## Measurements

### FPS (Frames Per Second)
Measures how fast the pipeline runs. Higher is better. Record the average FPS over the duration of the experiment. On constrained hardware (e.g. Jetson Orin Nano) this is especially important.

### Detection Accuracy — TP / TN / FP / FN

The detection task is a binary classification: for each object presented, the system either flags it as **unknown** or leaves it as **known**. The four possible outcomes are:

| Outcome | Meaning |
|---------|---------|
| **TP** (True Positive) | Unknown object correctly flagged as unknown |
| **TN** (True Negative) | Known object correctly left as known |
| **FP** (False Positive) | Known object incorrectly flagged as unknown |
| **FN** (False Negative) | Unknown object missed — incorrectly left as known |

From these, compute:

- **Precision** = TP / (TP + FP) — of everything flagged unknown, how much was actually unknown
- **Recall** = TP / (TP + FN) — of all actual unknowns, how many were caught
- **F1** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean of the two

For navigation safety, **recall matters more than precision** — missing a real unknown (FN) is more dangerous than a false alarm (FP).

---

## Experiments

| Experiment | Detection Method | Multimodal Model |
|------------|-----------------|-----------------|
| `experiment1.py` | YOLOv8s-worldv2 low-confidence threshold | Gemini 2.5 Flash Lite |

---

## Adding a New Experiment

A new experiment replaces Stage 2 and/or Stage 4 while keeping the rest of the pipeline identical:

- **Stage 2** — swap in a different detection method (e.g. NanoOWL/YOLO consensus, colour histogram change, custom model)
- **Stage 4** — swap in a different VLM (e.g. OpenAI GPT-4o, LLaVA, Moondream)
- Stages 1, 3, and 5 remain structurally the same across all experiments
