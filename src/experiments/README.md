Webcam (1280×720)
        │
        ▼
  Read frame (OpenCV)
        │
        ▼
  YOLOv8s-worldv2
        │
        ├── conf < 25%  ──────────────────────────────► Discard (noise, no box drawn)
        │
        ├── 25% ≤ conf < 50%  ───► Orange box on frame
        │                          "? <label> <conf>%"
        │                          unknown_found = True
        │                          snapshot = frame.copy()
        │                                │
        │                          streak counter ++
        │                                │
        │                         streak ≥ 8 frames?
        │                          │            │
        │                         No           Yes
        │                          │            │
        │                    "Candidate      "CONFIRMED UNKNOWN"
        │                     unknown..."     red banner on frame
        │                                          │
        │                                   cooldown elapsed?
        │                                    │           │
        │                                   No          Yes
        │                                    │           │
        │                              show timer    send snapshot
        │                                            to Gemini API
        │                                                │
        │                                    Gemini returns JSON
        │                                  { traversable, justification }
        │                                                │
        │                              ┌─────────────────┴──────────────────┐
        │                           true                                  false
        │                              │                                     │
        │                    "TRAVERSABLE" (green)              "NOT TRAVERSABLE" (red)
        │                    + justification text               + justification text
        │
        └── conf ≥ 50%  ──────────────► Blue box on frame
                                         "<label> <conf>%"
                                         streak counter = 0
        │
        ▼
  Display annotated frame
        │
        ▼
  ESC pressed? ──► Yes ──► Release camera, destroy windows, exit
        │
        No
        │
        └──────────────────────────────────────────────► next frame
