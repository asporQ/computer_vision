# track corner or point then create the tracking line

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
_, frame = cap.read()
frame = cv2.flip(frame,1)
prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ptmax = 30
# find corner
p0 = cv2.goodFeaturesToTrack(prvs,
                             mask=None,
                             maxCorners=ptmax,
                             qualityLevel=0.01,
                             minDistance=5,
                             blockSize=9)

line = np.zeros_like(frame).astype('uint8')
color = np.random.randint(0,255,(ptmax,3))

while(1):
    _,frame = cap.read()
    frame = cv2.flip(frame, 1)
    current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=prvs,
                                           nextImg=current,
                                           prevPts=p0,
                                           nextPts=None,
                                           winSize=(15,15),
                                           maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    good_new = p1[st==1] # เอาเฉพาะจุดที่เจอ ถ้าไม่เจอแล้วจะหาย -> prev point = now point 
    good_old = p0[st==1]

    for i in range(0,len(good_new)):
        try:
            line = cv2.line(line, (int(good_new[i,0]),int(good_new[i,1])), (int(good_old[i,0]),int(good_old[i,1])), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(good_new[i,0]),int(good_new[i,1])), 5, color[i].tolist(), -1)
        except:
            pass

    cv2.imshow('result',cv2.add(frame,line))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prvs = current.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
