import sys
import numpy as np
import cv2
import pylab as plt

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
imgL = np.zeros((480,640,3), np.uint8)
imgR = np.zeros((480,640,3), np.uint8)
while True:
    capL.read(imgL)
    capR.read(imgR)
    imgGrayL = cv2.GaussianBlur(
        cv2.equalizeHist(
            cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)),
        (5,5), 0)
    imgGrayR = cv2.GaussianBlur(
        cv2.equalizeHist(
            cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)),
        (5,5), 0)
    # cv2.imshow("image left gray", imgGrayL)
    # cv2.imshow("image right gray", imgGrayR)
    cv2.imshow("image left gray", imgL)
    cv2.imshow("image right gray", imgR)
    k = cv2.waitKey(33)
    if k == ord('d'):
        stereo = cv2.StereoBM(
            cv2.StereoBM_create(),
            ndisparities=int(sys.argv[3]),
            SADWindowSize=int(sys.argv[4]))
        disparity = stereo.compute(
            imgGrayL, imgGrayR)
        plt.imshow(disparity, "gray")
        plt.show()
    else:
        if k == ord('q'):
            break;