"""
mainDemo1.py
============
Pipeline 1 — Unknown-object detection via low-confidence YOLO scores
             + Gemini traversability assessment.

Flow
----
1. Open a live OpenCV webcam stream.
2. On every frame, run YOLOv8s-worldv2 (ObjectRecognition logic).
   Any detection whose confidence is below CONFIDENCE_BENCHMARK is
   treated as an *unknown object*.
3. When an unknown object is found, capture a screenshot of the frame
   and send it to Gemini (gemini-2.5-flash-lite) for traversability
   analysis.
4. Overlay all relevant information on the live stream:
     - Bounding boxes (blue = known, red = unknown)
     - Confidence labels / "UNKNOWN OBJECT" text
     - Latest Gemini verdict (traversable / not traversable + justification)
     - Cooldown timer so the API is not hammered every frame

Press ESC to quit.

Requirements
------------
    pip install ultralytics opencv-python google-generativeai pillow python-dotenv
    GEMINI_API_KEY must be set in a .env file at the project root (or as an
    environment variable).
"""

import os
import sys
import time
from pathlib import Path

import cv2
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # one level up from src/

YOLO_MODEL_PATH = SCRIPT_DIR / "unknown_object_detection" / "yolov8s-worldv2.pt"

CONFIDENCE_BENCHMARK = 50      # percent — below this → candidate unknown
MIN_DETECTION_CONF = 25        # percent — ignore detections below this entirely (noise filter)
UNKNOWN_PERSIST_FRAMES = 8     # object must be unknown for this many consecutive frames to trigger Gemini

# How many seconds to wait between Gemini API calls (avoid rate-limiting)
GEMINI_COOLDOWN_SECONDS = 5.0

OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICKNESS = 2
STATUS_SCALE = 0.55
STATUS_THICKNESS = 1

COLOR_KNOWN = (255, 0, 0)       # blue  — known object
COLOR_UNKNOWN = (0, 0, 255)     # red   — unknown object
COLOR_TRAVERSABLE = (0, 200, 0) # green — traversable verdict
COLOR_BLOCKED = (0, 0, 255)     # red   — not traversable
COLOR_PENDING = (200, 200, 0)   # yellow — waiting / no verdict yet


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    """Walk up the directory tree looking for a .env with GEMINI_API_KEY."""
    for directory in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        env_file = directory / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            break

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY not found. "
            "Add it to your .env file or set it as an environment variable."
        )
        sys.exit(1)
    return api_key


def query_gemini(model: genai.GenerativeModel, frame_bgr) -> dict:
    """
    Send a BGR OpenCV frame to Gemini and return a traversability verdict.

    Returns a dict with keys:
        traversable (bool), description (str), justification (str)
    """
    # Convert BGR → RGB → PIL
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    prompt = (
        "An autonomous robot has detected an unknown object in its path. "
        "Analyse this image and decide whether the robot can safely continue. "
        "Reply with a JSON object with exactly these fields:\n"
        '- "description": one concise sentence describing the scene\n'
        '- "traversable": true or false — can the robot safely pass?\n'
        '- "justification": one sentence explaining the decision\n'
        "Return only the raw JSON, no markdown fences."
    )

    import json
    response = model.generate_content([prompt, pil_image])
    raw = response.text.strip()

    # Strip markdown fences if the model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    return {
        "traversable": bool(parsed.get("traversable", False)),
        "description": parsed.get("description", ""),
        "justification": parsed.get("justification", ""),
    }


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def draw_status_line(frame, text: str, color, line_number: int):
    """Draw a status line in the top-left corner."""
    y = 28 + line_number * 24
    cv2.putText(frame, text, (12, y), OVERLAY_FONT, STATUS_SCALE, color, STATUS_THICKNESS)


def draw_verdict_banner(frame, verdict: dict | None, cooldown_remaining: float):
    """Draw the Gemini verdict at the bottom of the frame."""
    h, w = frame.shape[:2]

    if verdict is None:
        msg = "Gemini: no unknown object detected yet"
        cv2.putText(frame, msg, (12, h - 12), OVERLAY_FONT, STATUS_SCALE, COLOR_PENDING, STATUS_THICKNESS)
        return

    color = COLOR_TRAVERSABLE if verdict["traversable"] else COLOR_BLOCKED
    label = "TRAVERSABLE" if verdict["traversable"] else "NOT TRAVERSABLE"

    cv2.putText(frame, f"Gemini: {label}", (12, h - 48), OVERLAY_FONT, STATUS_SCALE, color, STATUS_THICKNESS)
    cv2.putText(frame, verdict["justification"][:90], (12, h - 24), OVERLAY_FONT, 0.45, color, 1)

    if cooldown_remaining > 0:
        cd_text = f"Next query in {cooldown_remaining:.1f}s"
        cv2.putText(frame, cd_text, (w - 220, h - 12), OVERLAY_FONT, 0.45, COLOR_PENDING, 1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

    model = YOLO(str(YOLO_MODEL_PATH))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")

    # Request a full-size resolution — camera will use the closest it supports
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    # Make the display window resizable so you can drag it larger if needed
    cv2.namedWindow("mainDemo1 — YOLO low-confidence + Gemini", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mainDemo1 — YOLO low-confidence + Gemini", actual_w, actual_h)

    print("mainDemo1 running — press ESC to quit.")
    print(f"Unknown-object threshold: confidence < {CONFIDENCE_BENCHMARK}%")
    print(f"Gemini cooldown: {GEMINI_COOLDOWN_SECONDS}s between queries")

    last_gemini_time = 0.0
    last_verdict: dict | None = None
    unknown_streak = 0  # consecutive frames with an unknown detection

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame.")
            continue

        results = model(frame, verbose=False, agnostic_nms=True, iou=0.6)

        unknown_found = False
        unknown_frame_snapshot = None  # frame to send to Gemini

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                conf_pct = conf * 100

                # Ignore very low-confidence detections entirely — they're noise
                if conf_pct < MIN_DETECTION_CONF:
                    continue

                label = result.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if conf_pct >= CONFIDENCE_BENCHMARK:
                    # Known object — blue box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_KNOWN, 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf_pct:.1f}%",
                        (x1, max(20, y1 - 10)),
                        OVERLAY_FONT, LABEL_SCALE, COLOR_KNOWN, LABEL_THICKNESS,
                    )
                else:
                    # Low-confidence detection — candidate unknown, draw in orange
                    unknown_found = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PENDING, 2)
                    cv2.putText(
                        frame,
                        f"? {label} {conf_pct:.1f}%",
                        (x1, max(20, y1 - 10)),
                        OVERLAY_FONT, LABEL_SCALE, COLOR_PENDING, LABEL_THICKNESS,
                    )
                    unknown_frame_snapshot = frame.copy()

        # Update streak counter
        if unknown_found:
            unknown_streak += 1
        else:
            unknown_streak = 0

        # Only treat as confirmed unknown after persisting for enough frames
        confirmed_unknown = unknown_streak >= UNKNOWN_PERSIST_FRAMES
        if confirmed_unknown and unknown_frame_snapshot is not None:
            # Redraw the box in red now that it's confirmed
            cv2.putText(
                frame, "UNKNOWN OBJECT",
                (12, 80), OVERLAY_FONT, LABEL_SCALE, COLOR_UNKNOWN, LABEL_THICKNESS,
            )

        # Query Gemini if confirmed unknown and cooldown has elapsed
        now = time.perf_counter()
        cooldown_remaining = max(0.0, GEMINI_COOLDOWN_SECONDS - (now - last_gemini_time))

        if confirmed_unknown and cooldown_remaining == 0.0 and unknown_frame_snapshot is not None:
            draw_status_line(frame, "Querying Gemini...", COLOR_PENDING, 0)
            cv2.imshow("mainDemo1 — YOLO low-confidence + Gemini", frame)
            cv2.waitKey(1)

            try:
                last_verdict = query_gemini(gemini_model, unknown_frame_snapshot)
                last_gemini_time = time.perf_counter()
                cooldown_remaining = GEMINI_COOLDOWN_SECONDS
                print(
                    f"[Gemini] traversable={last_verdict['traversable']} | "
                    f"{last_verdict['justification']}"
                )
            except Exception as e:
                print(f"[Gemini] query failed: {e}")

        # Status overlay (top-left)
        if confirmed_unknown:
            status_color = COLOR_UNKNOWN
            status_text = f"CONFIRMED UNKNOWN ({unknown_streak} frames)"
        elif unknown_found:
            status_color = COLOR_PENDING
            status_text = f"Candidate unknown... ({unknown_streak}/{UNKNOWN_PERSIST_FRAMES} frames)"
        else:
            status_color = COLOR_KNOWN
            status_text = "No unknown objects"
        draw_status_line(frame, status_text, status_color, 0)
        draw_status_line(frame, f"Confidence threshold: {CONFIDENCE_BENCHMARK}% | Min detection: {MIN_DETECTION_CONF}%", (200, 200, 200), 1)

        # Verdict banner (bottom)
        draw_verdict_banner(frame, last_verdict, cooldown_remaining)

        cv2.imshow("mainDemo1 — YOLO low-confidence + Gemini", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
