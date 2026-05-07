"""
mainDemo2.py
============
Pipeline 2 — Unknown-object detection via Nano-Owl / YOLO-World consensus
             + Gemini traversability assessment.

Flow
----
1. Open a live OpenCV webcam stream.
2. On every frame (Nano-Owl every N frames for speed), run the consensus
   detector from nano_owl_vs_yolo_world:
     - Nano-Owl detects a region but YOLO-World has no overlapping match
       → flagged as *unknown* (red box)
     - Nano-Owl detects a region and YOLO-World matches with LOW confidence
       → flagged as *weak match* (yellow box)
     - Both agree with high confidence → *known* (green box)
3. When at least one unknown (or weak-match) object is found, capture a
   screenshot and send it to Gemini (gemini-2.5-flash-lite) for
   traversability analysis.
4. Overlay all relevant information on the live stream:
     - Bounding boxes colour-coded by consensus result
     - YOLO-World detections (orange)
     - Latest Gemini verdict + justification
     - Cooldown timer

Press q to quit.

Requirements
------------
    pip install ultralytics nanoowl opencv-python google-generativeai pillow python-dotenv
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
from nanoowl.owl_predictor import OwlPredictor
from ultralytics import YOLOWorld

# Re-use the consensus helpers from the existing module
sys.path.insert(0, str(Path(__file__).resolve().parent / "unknown_object_detection"))
from nano_owl_vs_yolo_world import (
    YOLO_WORLD_MODEL,
    YOLO_WORLD_CLASSES,
    NANO_OWL_PRIMARY_PROMPT,
    NANO_OWL_FALLBACK_PROMPT,
    NANO_OWL_FRAME_INTERVAL,
    INFERENCE_MAX_WIDTH,
    IOU_MATCH_THRESHOLD,
    YOLO_WORLD_LOW_CONF_THRESHOLD,
    run_nano_owl,
    run_yolo_world,
    draw_detection,
    draw_status,
    resize_keep_aspect,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# How many seconds to wait between Gemini API calls
GEMINI_COOLDOWN_SECONDS = 5.0

# Scrutiny thresholds — tune these to adjust sensitivity
# Hard unknown (no YOLO match at all) must persist this many frames
UNKNOWN_PERSIST_FRAMES = 8
# Weak match (low-confidence YOLO) must persist longer before triggering Gemini
WEAK_PERSIST_FRAMES = 15
# Nano-Owl score below this is ignored entirely (noise filter)
# The imported NANO_OWL_THRESHOLD is 0.08 — we override it here to be stricter
NANO_OWL_MIN_SCORE = 0.18

OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_SCALE = 0.55
STATUS_THICKNESS = 1

COLOR_YOLO = (0, 165, 255)      # orange — YOLO-World detections
COLOR_UNKNOWN = (0, 0, 255)     # red    — unknown (no YOLO match)
COLOR_WEAK = (0, 255, 255)      # yellow — weak YOLO match
COLOR_KNOWN = (0, 255, 0)       # green  — both models agree
COLOR_TRAVERSABLE = (0, 200, 0) # green  — traversable verdict
COLOR_BLOCKED = (0, 0, 255)     # red    — not traversable
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
    import json

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

    response = model.generate_content([prompt, pil_image])
    raw = response.text.strip()

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

def draw_verdict_banner(frame, verdict: dict | None, cooldown_remaining: float):
    """Draw the Gemini verdict at the bottom of the frame."""
    h, w = frame.shape[:2]

    if verdict is None:
        cv2.putText(
            frame, "Gemini: no unknown object detected yet",
            (12, h - 12), OVERLAY_FONT, STATUS_SCALE, COLOR_PENDING, STATUS_THICKNESS,
        )
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

    # Load Nano-Owl
    nano_owl = OwlPredictor("google/owlvit-base-patch32", device="cpu")
    primary_text_encodings = nano_owl.encode_text(NANO_OWL_PRIMARY_PROMPT)
    fallback_text_encodings = nano_owl.encode_text(NANO_OWL_FALLBACK_PROMPT)

    # Load YOLO-World
    yolo_world_model_path = str(
        Path(__file__).resolve().parent / "unknown_object_detection" / YOLO_WORLD_MODEL
    )
    yolo_world = YOLOWorld(yolo_world_model_path)
    yolo_world.set_classes(YOLO_WORLD_CLASSES)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")

    print("mainDemo2 running — press q to quit.")
    print("Detection method: Nano-Owl / YOLO-World consensus")
    print(f"Gemini cooldown: {GEMINI_COOLDOWN_SECONDS}s between queries")

    frame_index = 0
    nano_detections = []
    nano_prompt_used = NANO_OWL_PRIMARY_PROMPT[0]

    last_gemini_time = 0.0
    last_verdict: dict | None = None
    unknown_streak = 0   # consecutive frames with a hard unknown (no YOLO match)
    weak_streak = 0      # consecutive frames with only weak YOLO matches

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame.")
            continue

        frame = resize_keep_aspect(frame, INFERENCE_MAX_WIDTH)

        # Run Nano-Owl every N frames (expensive)
        if frame_index % NANO_OWL_FRAME_INTERVAL == 0:
            nano_detections, nano_prompt_used = run_nano_owl(
                nano_owl, frame, primary_text_encodings, fallback_text_encodings,
            )

        yolo_detections = run_yolo_world(yolo_world, frame)

        # Draw YOLO-World detections (orange)
        for det in yolo_detections:
            draw_detection(frame, det, COLOR_YOLO)

        # Consensus classification
        unknown_count = 0
        matched_count = 0
        weak_match_count = 0
        unknown_frame_snapshot = None

        for nano_det in nano_detections:
            # Filter out low-scoring Nano-Owl detections — they're noise
            if nano_det.score < NANO_OWL_MIN_SCORE:
                continue

            # Find best overlapping YOLO detection
            best_match = None
            best_iou = 0.0
            for yolo_det in yolo_detections:
                ax1, ay1, ax2, ay2 = nano_det.box
                bx1, by1, bx2, by2 = yolo_det.box
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_match = yolo_det

            if best_match is None or best_iou < IOU_MATCH_THRESHOLD:
                # No YOLO match → hard unknown candidate
                unknown_count += 1
                draw_detection(frame, nano_det, COLOR_UNKNOWN)
                cv2.putText(
                    frame, f"? no YOLO match (score {nano_det.score:.2f})",
                    (nano_det.box[0], min(frame.shape[0] - 10, nano_det.box[3] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_UNKNOWN, 2,
                )
                unknown_frame_snapshot = frame.copy()

            elif best_match.score < YOLO_WORLD_LOW_CONF_THRESHOLD:
                # Weak YOLO match → uncertain candidate
                weak_match_count += 1
                draw_detection(frame, nano_det, COLOR_WEAK)
                cv2.putText(
                    frame,
                    f"? weak: {best_match.label} {best_match.score:.2f}",
                    (nano_det.box[0], min(frame.shape[0] - 10, nano_det.box[3] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WEAK, 2,
                )
                if unknown_frame_snapshot is None:
                    unknown_frame_snapshot = frame.copy()

            else:
                # Both models agree → known
                matched_count += 1
                draw_detection(frame, nano_det, COLOR_KNOWN)
                cv2.putText(
                    frame,
                    f"matched: {best_match.label} {best_match.score:.2f}",
                    (nano_det.box[0], min(frame.shape[0] - 10, nano_det.box[3] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_KNOWN, 2,
                )

        # Update streak counters
        if unknown_count > 0:
            unknown_streak += 1
        else:
            unknown_streak = 0

        if weak_match_count > 0 and unknown_count == 0:
            weak_streak += 1
        else:
            weak_streak = 0

        # Confirmed unknown: hard unknown persisted long enough
        confirmed_unknown = unknown_streak >= UNKNOWN_PERSIST_FRAMES
        # Confirmed weak: only weak matches but they've persisted even longer
        confirmed_weak = weak_streak >= WEAK_PERSIST_FRAMES

        should_query = (confirmed_unknown or confirmed_weak) and unknown_frame_snapshot is not None

        # Query Gemini if confirmed and cooldown elapsed
        now = time.perf_counter()
        cooldown_remaining = max(0.0, GEMINI_COOLDOWN_SECONDS - (now - last_gemini_time))

        if should_query and cooldown_remaining == 0.0:
            draw_status(frame, "Querying Gemini...", COLOR_PENDING, 4)
            cv2.imshow("mainDemo2 — Nano-Owl/YOLO consensus + Gemini", frame)
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

        # Status overlay (top-left) — mirrors nano_owl_vs_yolo_world style
        draw_status(frame, f"Nano-Owl prompt: {nano_prompt_used} | min score: {NANO_OWL_MIN_SCORE}", COLOR_KNOWN, 0)
        draw_status(frame, f"Nano-Owl refresh: every {NANO_OWL_FRAME_INTERVAL} frames", (255, 255, 255), 1)
        draw_status(frame, f"Matched: {matched_count} | Weak: {weak_match_count} (streak {weak_streak}/{WEAK_PERSIST_FRAMES})", COLOR_YOLO, 2)
        if confirmed_unknown:
            draw_status(frame, f"CONFIRMED UNKNOWN ({unknown_streak} frames)", COLOR_UNKNOWN, 3)
        elif unknown_count > 0:
            draw_status(frame, f"Unknown candidate... ({unknown_streak}/{UNKNOWN_PERSIST_FRAMES} frames)", COLOR_UNKNOWN, 3)
        else:
            draw_status(frame, f"Unknown candidates: 0", COLOR_KNOWN, 3)

        # Verdict banner (bottom)
        draw_verdict_banner(frame, last_verdict, cooldown_remaining)

        cv2.imshow("mainDemo2 — Nano-Owl/YOLO consensus + Gemini", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting.")
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
