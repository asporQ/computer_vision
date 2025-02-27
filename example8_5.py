import numpy as np
import cv2
import random

##### HOMOGRAPHIC TRANSFORMATION #####
# Create function to detect objects in video and surround them with rectangles of different colors

cap = cv2.VideoCapture(0)

detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

ref1 = cv2.imread('C:\\Users\\Nitro5\\OneDrive\\Documents\\CMU document\\cv\\computer_vision\\jklds.jpg', cv2.COLOR_BGR2GRAY)
ref2 = cv2.imread('C:\\Users\\Nitro5\\OneDrive\\Documents\\CMU document\\cv\\computer_vision\\WIN_20250224_10_51_28_Pro.jpg', cv2.COLOR_BGR2GRAY)

h1,w1,_ = ref1.shape
ref1 = cv2.resize(ref1,(int(w1*1.2),int(h1*1.2)))

h2,w2,_ = ref2.shape
ref2 = cv2.resize(ref2,(int(w2*1.2),int(h2*1.2)))

references = [(ref1, detector.detectAndCompute(ref1, None), (0, 0, 255)),(ref2, detector.detectAndCompute(ref2, None), (255, 0, 0))]  # Blue for ref2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = detector.detectAndCompute(gray, None)
    
    for ref, (kp1, des1), color in references:
        if des2 is not None and len(des2) > 0:
            matches = matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good) > 12:
                ref_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                target_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(ref_pts, target_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w, _ = ref.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    frame = cv2.polylines(frame, [np.int32(dst)], True, color, 3, cv2.LINE_AA)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
