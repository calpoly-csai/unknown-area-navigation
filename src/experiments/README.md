# Experiments

## Standard Pipeline

All experiments in this folder follow the same abstract pipeline. What varies between experiments is the **unknown object detection method** (Stage 2) and the **multimodal model** (Stage 4).

```
┌─────────────────────────────────────────────────────┐
│  Stage 1 — Capture                                  │
│                                                     │
│  Open webcam stream via OpenCV                      │
│  Read frame continuously in a loop                  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Unknown Object Detection                 │
│                                                     │
│  Run detection method on current frame              │
│  Classify each detected region as:                  │
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
               │  │  before dropping it;  this is a    │
               │  │  glitch &not a real detection.     │
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

```


