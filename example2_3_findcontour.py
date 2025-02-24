import numpy as np
import cv2

#For more info about contours: https://docs.opencv.org/4.5.3/d3/d05/tutorial_py_table_of_contents_contours.html

im = np.zeros(shape =(400,400)).astype('uint8')
polygons = [np.array([[20,20],[120,50],[30,80]]),
            np.array([[200,200],[200,350],[350,350], [350,200] ]),
            np.array([[170, 350], [60, 170], [130, 140]])]
colors = [255,255,255]
cv2.fillPoly(im,polygons,colors)
im[250:340,250:340] = 0
im[270:300,290:320] = 255

# contour = edges
# cv2.RETR_EXTERNAL : contour of the external line obj -> 3
# cv2.CHAIN_APPROX_NONE : collecting every pixel on contour
# cv2.CHAIN_APPROX_SIMPLE : collecting vertex of polygon
contours,hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.RETR_TREE : contour of every obj -> 5
# contours,hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#try mode cv2.RETR_EXTERNAL and cv2.RETR_TREE
#try method cv2.CHAIN_APPROX_NONE and cv2.CHAIN_APPROX_SIMPLE


print('Types of contours:', type(contours))
print('Length of contours:', len(contours))

# n is the number of contour of obj
# (a, b, c) a = number of pixel, b = x axis of pixel (column), c = y axis of pixel(row)
n = 0
print('Types of each contour:', type(contours[n]))
print('Shape of each contour:', contours[n].shape)
# print('Each contour value (list of points):\n',contours[n])

# create picture 
out = np.zeros(shape =(400,400)).astype('uint8')
for pos in contours[n]:
    out[pos[0,1],pos[0,0]] = 255


#print('Hierarchy:\n', 'Left Right Child Parent\n', hierarchy)


cv2.imshow('im', im)
cv2.imshow('out', out)


cv2.waitKey()
