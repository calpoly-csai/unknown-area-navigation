# Demo Pipeline 1: YOLO Low-Confidence + Gemini Traversability

## Overview

`mainDemo1.py` (and its Jetson-targeted copy `mainDemo1Jetson.py`) implements the first end-to-end unknown-object navigation pipeline. It combines a single-model object detector with a cloud vision-language model to make real-time traversability decisions from a live webcam stream.

The core idea: if a well-trained object detector sees something but can't confidently identify it, that object is a candidate unknown. If it persists in the frame long enough to rule out noise, a multimodal model is asked whether the robot can safely continue past it.

---

## Pipeline Architecture

```
Webcam frame
     │
     ▼
YOLOv8s-worldv2
     │
     ├── conf ≥ 50%  →  Known object  (blue box, label + confidence)
     ├── 25% ≤ conf < 50%  →  Candidate unknown  (orange box, streak counter++)
     └── conf < 25%  →  Ignored (noise)
                              │
                    streak ≥ 8 consecutive frames?
                              │
                    Yes → Confirmed unknown (red banner)
                              │
                    Gemini cooldown elapsed?
                              │
                    Yes → Send frame snapshot to Gemini 2.5 Flash Lite
                              │
                    Gemini returns: traversable / not traversable + justification
                              │
                    Verdict displayed on stream (bottom banner)
```

---

## Detection Logic

Unknown-object detection is based on **low confidence scores** from YOLOv8s-worldv2. The model is an open-vocabulary detector — it will attempt to classify anything it sees. When it detects a region but assigns low confidence, that indicates the object doesn't closely match any known category.

### Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_DETECTION_CONF` | 25% | Detections below this are silently dropped (noise filter) |
| `CONFIDENCE_BENCHMARK` | 50% | Below this but above min → candidate unknown |
| `UNKNOWN_PERSIST_FRAMES` | 8 frames | Consecutive frames required to confirm an unknown |
| `GEMINI_COOLDOWN_SECONDS` | 5.0s | Minimum gap between Gemini API calls |

### Three-State Classification

- **Known** (blue box): YOLO confidence ≥ 50%. Object is identified, no action taken.
- **Candidate unknown** (orange box): YOLO confidence between 25–50%. Streak counter increments each frame. Label shown as `? <class> <conf>%`.
- **Confirmed unknown** (red banner): Streak has reached 8 consecutive frames. Gemini is queried on the next available cooldown window.

The streak counter resets to zero as soon as no candidate unknown is detected in a frame. This means a person briefly walking through the frame will not trigger Gemini — only objects that remain in the scene persistently will.

---

## Multimodal Assessment

When a confirmed unknown is detected, a snapshot of the current frame is sent to **Gemini 2.5 Flash Lite** via the Google Generative AI API.

**Prompt sent to Gemini:**
> *"An autonomous robot has detected an unknown object in its path. Analyse this image and decide whether the robot can safely continue. Reply with a JSON object with exactly these fields: `description`, `traversable` (true/false), `justification`. Return only the raw JSON, no markdown fences."*

**Response fields used:**
- `traversable` — binary decision displayed as `TRAVERSABLE` (green) or `NOT TRAVERSABLE` (red)
- `justification` — one-sentence explanation shown in the bottom banner
- `description` — captured in the response dict but not currently displayed on stream

The last verdict persists on screen until a new query completes. The cooldown timer is shown in the bottom-right corner.

---

## Stream Overlay

| Element | Position | Colour | Description |
|---------|----------|--------|-------------|
| Detection box | Around object | Blue / Orange / Red | Known / candidate / confirmed unknown |
| Detection label | Above box | Matches box | Class name + confidence, or `? class conf%` |
| Status line 0 | Top-left | Varies | Current detection state + streak count |
| Status line 1 | Top-left | Grey | Active threshold values |
| `UNKNOWN OBJECT` banner | Top-left (y=80) | Red | Shown only when confirmed unknown is active |
| Gemini verdict | Bottom-left | Green / Red | `Gemini: TRAVERSABLE` or `NOT TRAVERSABLE` |
| Justification | Bottom-left | Matches verdict | Truncated to 90 characters |
| Cooldown timer | Bottom-right | Yellow | Seconds until next Gemini query is allowed |

---

## Platform Variants

### `mainDemo1.py` — macOS / Linux

- Intended for development and testing on a standard desktop or laptop.
- Uses `cv2.VideoCapture(0)` with a requested resolution of 1280×720.
- Window is created with `cv2.WINDOW_NORMAL` (resizable).
- Quit with **ESC**.

### `mainDemo1Jetson.py` — NVIDIA Jetson Orin Nano

- Functionally identical to `mainDemo1.py`.
- Separated to allow Jetson-specific dependency management without affecting the main codebase.
- Requires the JetPack-matched PyTorch wheel (`torch-2.3.0-cp310-cp310-linux_aarch64.whl`) and a compatible `torchvision` (0.18.0).
- Known environment issues on Jetson: `torchvision::nms does not exist` (torch/torchvision version mismatch) and `numpy not available RuntimeError` (numpy ≥ 2.0 incompatible with torch 2.3.0 — requires `numpy<2.0`).
- Quit with **ESC**.

---

## Running

```bash
# From the src/ directory
python mainDemo1.py          # macOS / Linux
python mainDemo1Jetson.py    # Jetson Orin Nano
```

Requires `GEMINI_API_KEY` in a `.env` file at the project root (or set as an environment variable).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8s-worldv2 inference |
| `opencv-python` | Webcam capture and stream overlay |
| `google-generativeai` | Gemini API client |
| `pillow` | Frame conversion for Gemini (BGR → RGB → PIL) |
| `python-dotenv` | `.env` file loading for API key |

---

## Known Limitations

- **Single-model detection**: relies entirely on YOLO confidence as a proxy for "unknown". Objects that YOLO happens to assign high confidence to (even if misidentified) will not be flagged.
- **No spatial filtering**: the streak counter is global — if multiple candidate unknowns appear and disappear across different regions of the frame, their counts are not tracked independently.
- **Gemini latency**: API calls take ~1–3s. During this time the stream continues but the verdict is stale. The cooldown prevents hammering but means the displayed verdict may lag behind the current scene.
- **Single camera**: only `cv2.VideoCapture(0)` is supported. No multi-camera or depth input.
