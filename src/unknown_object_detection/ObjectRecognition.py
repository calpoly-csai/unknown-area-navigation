import time

import cv2 #for computer vision stuff
from ultralytics import YOLO

def object_recognition():
    cap = cv2.VideoCapture(0)
    model = YOLO("YOLOv8s-worldv2.pt") #Presage
    confidencebenchmark = 50 #in percent

    while True:
        ret, frame = cap.read()
        results = model(frame, verbose=False, agnostic_nms=True, iou=0.6) #agnostic_nms= delete competing overlap if confidence lower
        #iou= how much overlap needed to 'kick out'

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0]) #class id
                conf = float(box.conf[0]) #confidence
                label = result.names[cls_id] #class name
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #get the x and y of the corners of the box
                text = f"{label} {conf * 100:.1f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) #draw a box
                if conf*100 > confidencebenchmark: #if confidence high
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3) #put the text above the box
                if conf*100 < confidencebenchmark: #if confidence low
                    cv2.putText(frame, "UNKNOWN OBJECT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # esc key
            print("exiting")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    object_recognition()