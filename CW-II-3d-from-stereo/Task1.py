import cv2
import numpy as np

dp = 1 # acc same size with image
minDist = 15
mag_thr = 40
acc_thr = 30

def detectCircle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp,
        minDist,
        param1=mag_thr,   # mag thr 
        param2=acc_thr,   # acc thr
        minRadius=12,
        maxRadius=35
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            cv2.circle(img, (x, y), r, (0,255,0), 2)


image0 = cv2.imread("./view0.png", 1)
image1 = cv2.imread("./view1.png", 1)
detectCircle(image0)
detectCircle(image1)
cv2.imwrite("image0.png", image0)
cv2.imwrite("image1.png", image1)