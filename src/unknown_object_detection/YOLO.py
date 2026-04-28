import cv2
from ultralytics import YOLO

# Load in an YOLO model (example)
model = YOLO("yolo11n.pt")  

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the current frame
    results = model(frame, verbose=False)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Show the live result
    cv2.imshow("Webcam + YOLO", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()