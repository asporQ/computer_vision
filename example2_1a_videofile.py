import cv2

#Download 'ExampleBGSubtraction.avi' from https://drive.google.com/file/d/1OD_A0wqN2Om2SusCztybu-_hMSUQuRt7/view?usp=sharing

cap = cv2.VideoCapture('C:\\Users\\Nitro5\\Downloads\\ExampleBGSubtraction.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

while(cap.isOpened()):
    # haveframe is to check is the frame exist?
    haveFrame, im = cap.read()

    # end loop if no anymore frame or key from video is q 
    # wait key is large number from the FPS longer
    if (not haveFrame) or (cv2.waitKey(int(1000/fps)) & 0xFF == ord('q')):
        break

    cv2.imshow('video',im)

cap.release()
cv2.destroyAllWindows()
