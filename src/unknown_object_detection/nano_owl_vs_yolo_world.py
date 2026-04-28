'''
Author: Ceya
Description: Compare Nano-Owl and YOLO-World on a live webcam feed.
Nano-Owl is prompted to detect generic objects while YOLO-World is asked
to classify likely everyday categories. If Nano-Owl detects an object and
YOLO-World returns no match or only weak confidence, the frame is flagged
as a possible unknown object scenario.
'''

from dataclasses import dataclass

import cv2
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor
from ultralytics import YOLOWorld


YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"
YOLO_WORLD_CLASSES = [
    "person",
    "cell phone",
    "remote",
    "book",
    "bottle",
    "cup",
    "mouse",
    "keyboard",
    "pen",
    "marker",
    "tool",
    "box",
]

# Nano-Owl examples use a prompt like "[an owl, a glove]". In the Python API
# the equivalent prompt is a list of labels, so ["objects"] mimics "[objects]".
NANO_OWL_PRIMARY_PROMPT = ["objects"]
NANO_OWL_FALLBACK_PROMPT = ["handheld object"]

NANO_OWL_THRESHOLD = 0.08
YOLO_WORLD_SHOW_THRESHOLD = 0.10
YOLO_WORLD_LOW_CONF_THRESHOLD = 0.35
IOU_MATCH_THRESHOLD = 0.20 # count as same object if overlap at least 20%
INFERENCE_MAX_WIDTH = 640 # for speed
NANO_OWL_FRAME_INTERVAL = 3 # for speed 
OVERLAY_FONT = cv2.FONT_HERSHEY_DUPLEX
STATUS_FONT_SCALE = 0.5
STATUS_FONT_THICKNESS = 1
LABEL_FONT_SCALE = 0.48
LABEL_FONT_THICKNESS = 1


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def resize_keep_aspect(frame, max_width):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))


def extract_nano_owl_detections(output, prompts, width, height):
    boxes = getattr(output, "boxes", None)
    scores = getattr(output, "scores", None)
    labels = getattr(output, "labels", None)

    if boxes is None or scores is None or labels is None:
        return []

    detections = []
    for box, score, label_idx in zip(boxes, scores, labels):
        label_name = prompts[int(label_idx)]
        detections.append(
            Detection(
                label=label_name,
                score=float(score),
                box=clip_box(box, width, height),
            )
        )
    return detections


def run_nano_owl(
    predictor,
    frame_bgr,
    primary_text_encodings,
    fallback_text_encodings,
):
    height, width = frame_bgr.shape[:2]
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    primary_output = predictor.predict(
        image=image_pil,
        text=NANO_OWL_PRIMARY_PROMPT,
        text_encodings=primary_text_encodings,
        threshold=NANO_OWL_THRESHOLD,
    )
    primary_detections = extract_nano_owl_detections(
        primary_output,
        NANO_OWL_PRIMARY_PROMPT,
        width,
        height,
    )
    if primary_detections:
        return primary_detections, NANO_OWL_PRIMARY_PROMPT[0]

    fallback_output = predictor.predict(
        image=image_pil,
        text=NANO_OWL_FALLBACK_PROMPT,
        text_encodings=fallback_text_encodings,
        threshold=NANO_OWL_THRESHOLD,
    )
    fallback_detections = extract_nano_owl_detections(
        fallback_output,
        NANO_OWL_FALLBACK_PROMPT,
        width,
        height,
    )
    return fallback_detections, NANO_OWL_FALLBACK_PROMPT[0]


def run_yolo_world(model, frame_bgr):
    result = model.predict(frame_bgr, verbose=False)[0]
    detections = []
    boxes = result.boxes

    if boxes is None:
        return detections

    names = result.names
    # class id (cls_id) is index of class name
    for xyxy, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        score = float(conf)
        if score < YOLO_WORLD_SHOW_THRESHOLD:
            continue

        detections.append(
            Detection(
                label=names[int(cls_id)],
                score=score,
                box=tuple(int(v) for v in xyxy.tolist()),
            )
        )
    return detections


def draw_detection(frame, detection, color):
    x1, y1, x2, y2 = detection.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{detection.label}: {detection.score:.2f}",
        (x1, max(20, y1 - 8)),
        OVERLAY_FONT,
        LABEL_FONT_SCALE,
        color,
        LABEL_FONT_THICKNESS,
    )


def draw_status(frame, label, color, line_number):
    y = 24 + (line_number * 22)
    cv2.putText(
        frame,
        label,
        (12, y),
        OVERLAY_FONT,
        STATUS_FONT_SCALE,
        color,
        STATUS_FONT_THICKNESS,
    )


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

# returns the YOLO-World detection that overlaps with the Nano-Owl detection most
def match_nano_to_yolo(nano_detection, yolo_detections):
    best_match = None
    best_iou = 0.0

    for yolo_detection in yolo_detections:
        iou = compute_iou(nano_detection.box, yolo_detection.box)
        if iou > best_iou:
            best_iou = iou
            best_match = yolo_detection #best match to an object, labeled w the object name from YOLO

    if best_match is None:
        return None, 0.0
    return best_match, best_iou #means object was detected by nano but not YOLO = unknown object 


def main():
    nano_owl = OwlPredictor("google/owlvit-base-patch32", device="cpu")
    primary_text_encodings = nano_owl.encode_text(NANO_OWL_PRIMARY_PROMPT)
    fallback_text_encodings = nano_owl.encode_text(NANO_OWL_FALLBACK_PROMPT)

    yolo_world = YOLOWorld(YOLO_WORLD_MODEL)
    yolo_world.set_classes(YOLO_WORLD_CLASSES)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press q to quit.")
    print("Try a narrow or unusual item against a plain background: USB adapter,")
    print("marker cap, measuring tape, earbuds case, or small hobby tool.")

    frame_index = 0
    nano_detections = []
    nano_prompt_used = NANO_OWL_PRIMARY_PROMPT[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_keep_aspect(frame, INFERENCE_MAX_WIDTH)

        if frame_index % NANO_OWL_FRAME_INTERVAL == 0:
            nano_detections, nano_prompt_used = run_nano_owl(
                nano_owl,
                frame,
                primary_text_encodings,
                fallback_text_encodings,
            )
        yolo_detections = run_yolo_world(yolo_world, frame)

        for detection in yolo_detections:
            draw_detection(frame, detection, (0, 165, 255)) #orange-ish box around yolo detections

        unknown_count = 0
        matched_count = 0
        weak_match_count = 0

        for detection in nano_detections:
            matched_detection, best_iou = match_nano_to_yolo(detection, yolo_detections)

            if matched_detection is None or best_iou < IOU_MATCH_THRESHOLD:
                unknown_count += 1
                draw_detection(frame, detection, (0, 0, 255)) #red box around unknown objects 
                cv2.putText(
                    frame,
                    "unknown candidate",
                    (detection.box[0], min(frame.shape[0] - 10, detection.box[3] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
                continue

            if matched_detection.score < YOLO_WORLD_LOW_CONF_THRESHOLD:
                weak_match_count += 1
                draw_detection(frame, detection, (0, 255, 255)) #yellow box around weak known guess
                cv2.putText(
                    frame,
                    f"weak YOLO match: {matched_detection.label} {matched_detection.score:.2f}",
                    (detection.box[0], min(frame.shape[0] - 10, detection.box[3] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                continue

            matched_count += 1
            draw_detection(frame, detection, (0, 255, 0)) # green for matched in both
            cv2.putText(
                frame,
                f"matched: {matched_detection.label} {matched_detection.score:.2f}",
                (detection.box[0], min(frame.shape[0] - 10, detection.box[3] + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        # summary stats 
        draw_status(frame, f"Nano-Owl prompt: {nano_prompt_used}", (0, 255, 0), 0)
        draw_status(
            frame,
            f"Nano-Owl refresh: every {NANO_OWL_FRAME_INTERVAL} frames",
            (255, 255, 255),
            1,
        )
        draw_status(
            frame,
            f"Nano matched: {matched_count} | weak matches: {weak_match_count}",
            (0, 165, 255),
            2,
        )
        draw_status(frame, f"Unknown candidates: {unknown_count}", (0, 0, 255), 3)

        cv2.imshow("Nano-Owl vs YOLO-World", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
