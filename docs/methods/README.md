# Methods

This folder contains the methods documentation for each subsystem of the autonomous navigation supplementation pipeline. Each file corresponds to one team's work.

| File | Subsystem |
|---|---|
| [`unknown_object_detection.md`](./unknown_object_detection.md) | Detecting objects a model cannot confidently classify |
| [`multimodal_interpretation.md`](./multimodal_interpretation.md) | VLM-based scene description and traversability assessment |
| [`navigation.md`](./navigation.md) | AprilTag-based pose estimation and directional movement control |

---

## Study Overview

This project develops a modular pipeline to supplement autonomous navigation in unfamiliar or out-of-distribution environments. Standard deep learning models are trained on fixed datasets and fail to generalize when they encounter objects or scenes outside that distribution. The pipeline addresses this in two stages:

1. **Unknown Object Detection** — identify when a model is uncertain about what it sees and flag the object rather than silently mislabeling it.
2. **Multimodal Scene Interpretation** — use a vision-language model to generate a natural-language description of the scene and a binary traversability judgment.
3. **Navigation** — use AprilTag fiducial markers as visual landmarks to estimate 3D pose and issue directional movement commands in real time.

All three subsystems are exploratory prototypes evaluated on live webcam input and researcher-supplied test images.
