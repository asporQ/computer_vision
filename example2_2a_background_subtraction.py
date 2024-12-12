import cv2

#Download 'ExampleBGSubtraction.avi' from https://drive.google.com/file/d/1OD_A0wqN2Om2SusCztybu-_hMSUQuRt7/view?usp=sharing

cap = cv2.VideoCapture('C:\\Users\\Nitro5\\Downloads\\ExampleBGSubtraction.avi')

# assume first frame is no obj -> bg
haveFrame,bg = cap.read()

while(cap.isOpened()):
    haveFrame,im = cap.read()

    if (not haveFrame) or (cv2.waitKey(70) & 0xFF == ord('q')):
        break
    
    # |im - bg| (uint8) to find obj 
    # im = color img
    # bg = color img
    # RGB different  
    diffc = cv2.absdiff(im,bg)
    # BW different
    diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY)
    bwmask = cv2.inRange(diffg,50,255)

    # type is uint8
    # print(type(diffc[0, 0, 0]))
    # print(type(diffg[0, 0]))
    # print(type(bwmask[0, 0]))

    cv2.imshow('diffc', diffc)
    cv2.moveWindow('diffc',10,10)
    cv2.imshow('diffg',diffg)
    cv2.moveWindow('diffg', 400, 10)
    cv2.imshow('bwmask', bwmask)
    cv2.moveWindow('bwmask', 800, 10)

cap.release()
cv2.destroyAllWindows()
