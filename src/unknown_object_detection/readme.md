1. YOLO-WORLD Detection -> Detect Low Confidence -> Unknown Object Detected

**Description:** YOLO-World is an open-world object detector, so it can detect a very wide range of objects that appear regularly (chairs, people, etc…). When running with the OpenCV script, if the model outputs a confidence level below a certain threshold, we’ll classify it as unknown.

**Your Task:** Integrate the small YOLO-WORLD (YOLOv8s-worldv2)model into your OpenCV script (which opens your camera and streams). Add logic so that you can extract the confidence scores that the model produces when it sees various objects so that we can classify low confidence objects as “unknown”. For example, if you present a niche pen (or any niche object), the model should produce low confidence scores as it tries to guess what it is, leading us to classify it as unknown. 

---

2. **Class Agnostic Model + Traditional Object Detection Model -> If general object detected, but not detected by traditional model -> Unknown Object Detected**

**Description:** A class agnostic model just tries and detects whether an object is present or not, not going as far as to detect what an object is. A traditional object detection model regularly detects objects based on trained data. If our class agnostic model determines that an object is present, but our traditional object model cannot classify what it is, we’ll determine that it’s an unknown object. For example, say that we have our class agnostic model and an traditional model only trained on apples. If I present an shoebox in front of the camera, the class agnostic model will detect an object is there, but the traditional model won’t detect anything, leading us to conclude that it’s an unknown object.

**Your Task:**

Create a dataset for the class agnostic model. Take pictures of 20-30 objects in front of a pretty basic background (like a wall) and label those objects under the classname “Object” using the AnyLabeling service. Don’t take pictures of the same object many times – remember that we are training a model to detect if anything is present in front of the camera, rather than training it to classify a specific object. Finally, export the dataset through Roboflow, and I will train the model using my professor’s fancy computer and get it back to you to finish the pipeline.