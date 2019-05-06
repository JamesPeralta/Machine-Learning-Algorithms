import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

while(1):
    ret, frame = cap.read()
    frame = np.rot90(frame, 2)
    cv.imshow("FrameTing", frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
