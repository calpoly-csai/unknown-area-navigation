import cv2
import numpy as np
import glob

# IMPORTANT !!!!
# Ensure that the camera resolution is the same as those used in cvAprilTag.py for accurate pose estimation of AprilTags.

# Inner corners of the checkerboard pattern (number of squares - 1, in each dimension)
checkerboard = (8, 6)
square_size = 0.02  # meters

objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# Folder containing calibration images of the checkerboard pattern
images = glob.glob("src/navigation/calib_images/*.jpg")
print("Found", len(images), "images for calibration.")
if len(images) == 0:
    print("No images found. Add checkerboard photos to src/navigation/calib_images/ and re-run.")
    exit()

for fname in images:
    print("Processing", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, checkerboard, None)
    if ok:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("K=", K)
print("dist=", dist)
print("reprojection error=", ret)
print("camera params =", K[0,0], K[1,1], K[0,2], K[1,2])

#Save the camera parameters to a file for later use in cvAprilTag.py
np.savez("camera_params.npz", K=K, dist=dist)

# For my HP's webcam, I got:
# camera_params = [1361.83171, 1371.11287, 956.335161, 519.532480]