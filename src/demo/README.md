# Demo

## Pipeline

All demos follow this pipeline. The unknown object detection method and multimodal model vary between demos.

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
│  Classify objects as:                               │
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
               │  │  before dropping it — this is      │
               │  │  likely a glitch, not a real       │
               │  │  detection. The persistence gate   │
               │  │  filters these out by requiring    │
               │  │  the unknown to appear             │
               │  │  consistently before treating      │
               │  │  it as real.                       │
               │  │                                    │
               │  │  N frames elapsed?                 │
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
│  Stage 5 — Navigation (runs on every frame)         │
│                                                     │
│  AprilTag orientation is tracked continuously,      │
│  regardless of whether an unknown was detected.     │
│                                                     │
│  AprilTag pose estimation (cvAprilTag.py):          │
│      tag to the left  → Move Left                   │
│      tag to the right → Move Right                  │
│      no tag detected  → No Tag Detected             │
│                                                     │
│  Forward movement is gated by traversability:       │
│      traversable=true  + tag far  → Move Forward    │
│      traversable=true  + tag close → Stop           │
│      traversable=false (any dist) → Stop            │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 6 — Display                                  │
│                                                     │
│  Annotate frame with detection results              │
│  Show traversability verdict + justification        │
│  Show navigation direction                          │
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

## Navigation — cvAprilTag.py

Navigation is handled by detecting AprilTags in the frame and using their 3D pose to determine movement direction.

```python
# Movement control logic based on the tag's position (pose_t)
if tag.pose_t[0] < -0.075:      # Tag is to the left
    → Move Left

elif tag.pose_t[0] > 0.075:     # Tag is to the right
    → Move Right

elif tag.pose_t[2] > 0.1:       # Tag is far
    → Move Forward

elif tag.pose_t[2] <= 0.1:      # Tag is close
    → Stop

if len(tags) == 0:
    → No Tag Detected
```

Pose estimation requires camera calibration parameters `[fx, fy, cx, cy]` obtained from `calibration.py`. The tag size is set to `0.115m` (length of one side of the physical AprilTag square).

## Demos

| Demo | Detection Method | Multimodal Model |
|------|-----------------|-----------------|
| `mainDemo1.py` | YOLOv8s-worldv2 low-confidence threshold | Gemini 2.5 Flash Lite |
| `mainDemo2.py` | NanoOWL / YOLO-World consensus | Gemini 2.5 Flash Lite |
