#Download images from https://drive.google.com/file/d/1KqllafwQiJR-Ronos3N-AHNfnoBb8I7H/view?usp=sharing

import cv2

def coinCounting(filename):
    im = cv2.imread(filename)
    negative = 256 -1 - im
    cv2.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))


coinCounting('C:\\Users\\Nitro5\\OneDrive\\Documents\\CMU document\\cv\\homework\\CoinCounting\\coin1.png')
