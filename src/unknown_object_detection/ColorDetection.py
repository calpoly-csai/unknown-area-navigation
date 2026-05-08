import cv2
import time

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
previouscolor = cv2.calcHist([frame],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
previouscolor = cv2.normalize(previouscolor,previouscolor).flatten()
text_until = 0
while cap.isOpened():
    ret, frame = cap.read()
    currentcolor = cv2.calcHist([frame],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    currentcolor = cv2.normalize(currentcolor,currentcolor).flatten()
    diff = cv2.compareHist(previouscolor, currentcolor, cv2.HISTCMP_BHATTACHARYYA)
    print(diff, "is the difference between previous color and current color")

    if diff > 0.1:
        print("COLOR CHANGED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        text_until = time.time() + 0.5

    if time.time() < text_until:
        cv2.putText(frame, "Color Changed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow('webcam',frame)

    previouscolor = currentcolor

    if cv2.waitKey(1) & 0xFF == ord('q'):#checks for key 'q'
        break
cap.release()
cv2.destroyAllWindows()#end