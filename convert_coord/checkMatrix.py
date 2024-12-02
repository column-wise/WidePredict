import numpy as np
import cv2

oldMtx = np.load("Original camera matrix.npy")
coef = np.load("Distortion coefficients.npy")
newMtx = np.load("Optimal camera matrix.npy")
cam = cv2.VideoCapture(0)
(w, h) = (int(cam.get(4)), int(cam.get(3)))
while (True):
    _, frame = cam.read()
    undis = cv2.undistort(frame, oldMtx, coef, newMtx)
    cv2.imshow("Original vs Undistortion", np.hstack([frame, undis]))
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()