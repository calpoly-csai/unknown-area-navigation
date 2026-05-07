# Methods — AprilTag-Based Navigation

## Overview

Once an unknown object or non-traversable scene has been identified, an autonomous agent needs a reliable mechanism to orient itself and navigate toward a known waypoint. This subsystem explores using **AprilTag fiducial markers** as visual landmarks for real-time pose estimation and directional control. A camera detects a printed tag, estimates its 3D position relative to the vehicle, and issues movement commands based on that pose.

---

## Study Design

This is an **exploratory, systems-development study**. The approach was implemented as a live webcam prototype and evaluated qualitatively by placing an AprilTag at varying positions and distances from the camera and observing detection and directional output behavior. No standardized benchmark dataset was used in this phase.

---

## Task Definition

Given a live camera feed, the system must:

1. **Detect** any AprilTag present in the current frame
2. **Estimate pose** — recover the tag's 3D translation vector relative to the camera
3. **Issue a movement command** — one of `Move Left`, `Move Right`, `Move Forward`, or `Stop`, based on the tag's position
4. **Handle the no-tag case** — display `No Tag Detected` when no marker is visible

---

## Camera Calibration

Accurate pose estimation requires knowledge of the camera's intrinsic parameters. A separate calibration script (`calibration.py`) was developed to compute these before running the main detection loop.

### Procedure

1. A checkerboard calibration target with an **8×6 inner-corner grid** and a **20 mm square size** was used.
2. Multiple `.jpg` images of the checkerboard at varying angles and distances were collected and stored in a calibration images folder.
3. For each image:
   - The frame was converted to grayscale.
   - `cv2.findChessboardCorners` was used to locate the inner corners.
   - `cv2.cornerSubPix` refined corner positions to sub-pixel accuracy using a termination criterion of 30 iterations or ε = 0.001.
4. `cv2.calibrateCamera` was called with the collected object-point / image-point pairs to recover the **camera matrix K** and **distortion coefficients**.
5. The resulting parameters were saved to `camera_params.npz` for use in the detection script.

### Output

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

dist = [k1, k2, p1, p2, k3]
reprojection error = <float>
```

> **Note:** Camera resolution must match between calibration and detection. The calibration script includes a reminder comment to this effect. For the HP webcam used during development, the recovered parameters were `[fx=1361.83, fy=1371.11, cx=956.34, cy=519.53]`.

---

## AprilTag Detection and Pose Estimation

**Implementation:** `src/navigation/cvAprilTag.py`

**Libraries:** `pupil_apriltags`, OpenCV (`cv2`), NumPy

### Detection Pipeline

For each frame captured from the webcam:

1. The frame was captured at **1920×1080** resolution via `cv2.VideoCapture`.
2. The frame was converted to grayscale — AprilTag detection operates on single-channel images.
3. `pupil_apriltags.Detector` was used to detect all tags in the grayscale frame.
4. Pose estimation was enabled by passing the calibrated camera parameters and the physical tag size to the detector:

| Parameter | Value | Description |
|---|---|---|
| `camera_params` | `[1361.83, 1371.11, 956.34, 519.53]` | `[fx, fy, cx, cy]` from calibration |
| `tag_size` | `0.115 m` | Physical side length of the printed AprilTag |

5. For each detected tag, the detector returned:
   - `tag.corners` — four 2D corner points in image space
   - `tag.tag_id` — the integer ID encoded in the marker
   - `tag.pose_t` — 3D translation vector `[x, y, z]` in meters (camera frame)
   - `tag.pose_R` — 3×3 rotation matrix

### Coordinate Convention

The translation vector `pose_t` is expressed in the **camera coordinate frame**:

| Axis | Meaning |
|---|---|
| `pose_t[0]` (x) | Lateral offset — negative is left, positive is right |
| `pose_t[1]` (y) | Vertical offset |
| `pose_t[2]` (z) | Depth — distance from camera to tag along the optical axis |

---

## Movement Control Logic

Directional commands were derived from the tag's translation vector using fixed thresholds:

| Condition | Command | Display Color |
|---|---|---|
| `pose_t[0] < −0.075 m` | **Move Left** | White |
| `pose_t[0] > +0.075 m` | **Move Right** | White |
| `pose_t[2] > 0.1 m` | **Move Forward** | White |
| `pose_t[2] ≤ 0.1 m` | **Stop** | White |
| No tag detected | **No Tag Detected** | White |

The lateral threshold of ±7.5 cm defines a dead-band around center — within this band, the tag is considered aligned and the depth condition takes over. The depth threshold of 10 cm defines the stopping distance.

> **Note:** Conditions are evaluated in order. Lateral misalignment is checked before depth, so the agent corrects its heading before advancing.

---

## Visualization

For each detected tag, the following overlays were drawn on the grayscale frame:

- **Bounding box** — green polygon connecting the four tag corners (`cv2.polylines`)
- **Tag ID** — green text above the top-left corner
- **Pose vectors** — `pose_t` and `pose_R` values printed as blue text near the tag
- **Movement command** — white text in the top-left corner of the frame

---

## Study Materials

| Component | Details |
|---|---|
| Camera input | Live webcam feed (`cv2.VideoCapture(0)`) at 1920×1080 |
| Calibration target | 8×6 inner-corner checkerboard, 20 mm squares |
| AprilTag | Printed marker, 115 mm side length |
| Hardware | Standard laptop webcam; CPU inference |
| Frameworks | `pupil_apriltags`, OpenCV, NumPy |
| Output | Real-time annotated grayscale display with movement command overlay |

---

## Limitations

- A single tag is assumed per scene; multi-tag disambiguation logic was not implemented.
- Movement commands are discrete and threshold-based; no continuous control signal or PID controller was implemented.
- Pose estimation accuracy degrades at steep viewing angles and at distances where the tag subtends fewer pixels.
- The system operates on individual frames with no temporal smoothing — rapid tag movement can cause flickering command output.
- Calibration was performed for one specific webcam; parameters must be recollected for any different camera or resolution.
- No integration with the unknown object detection or multimodal interpretation subsystems was implemented in this phase.
