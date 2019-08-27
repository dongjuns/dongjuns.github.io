---
title: "Vehicle Number Plate Recognition"
date: 2019-08-26 17:04:00 +0900
categories: Machine Vision
---

## 0. Import Libraries
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

## 1. Load the image file
```
img = cv2.imread('car.jpg')
height, width, channel = img.shape # height x width x color channel, (960, 1280, 3)
plt.figure(figsize=(12,12))
plt.imshow(img)
```

## 2. Image Processing
(1) Convert color to gray     
(2) Threshold for image binarization (change the pixel to black or white)     
(3) Gaussian Blurring, Gaussian smoothing filtering     
```
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgThreshold = cv2.adaptiveThreshold(imgGray,
                                  maxValue=255.0,
                                  adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  thresholdType=cv2.THRESH_BINARY_INV,
                                  blockSize=19,
                                  C=9)
                                  
imgThresholdBlur = cv2.GaussianBlur(imgThreshold, ksize=(5,5), sigmaX=1)                         
```

*adaptiveMethod, T(x,y) is related to the blockSizeXblockSize - C      
-> ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C     

*threshholdType, THRESH_BINARY or THRESH_BINACRY_INV

*GaussianBlur is the kernel consists of pixel values of Gaussian distribution.

Build a mask (0, 0, 0, ..., 0) and extract the contours.      
It retrieves all the contours using the image of binary pixels.     
then draw all the contour as a box from saved contours in findContours.
```
imgMask = np.zeros((height, width, channel), dtype=np.uint8)

contours, _ = cv2.findContours(imgThresholdBlur,
                                  mode=cv2.RETR_LIST,
                                  method=cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgMask, contours=contours, contourIdx=-1, color=(255, 255, 255))

plt.imshow(imgMask)
```

We now have a many contours,      
and save all the contour to contours {} for box processing.
```
imgMask = np.zeros((height, width, channel), dtype=np.uint8)
contours = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(imgMask, pt1=(x, y), pt2=(x+width, y+height), color=(255, 255, 255), thickness=1)
    
    contours.append({
        'contour':contour,
        'w':w,
        'h':h,
        'x':x,
        'y':y,
        'cx':(w/2)+x,
        'cy':(h/2)+y
    })
    
plt.imshow(imgMask)
```

It's time to clean the boxes up.      
Think of size of box, 
```
minWidth, minHeight = 1, 12
minRatio, maxRatio = 0.5, 1.0

possibleContours = []

cnt = 0
for c in contours:
    area = c['w'] * c['h']
    ratio = c['w'] / c['h']
    
    if (c['w'] > minWidth) and (c['h'] > minHeight) and (minRatio < ratio < maxRatio):
        c['idx'] = cnt
        cnt += 1
        possibleContours.append(c)
        
imgMask = np.zeros((height, width, channel), dtype=np.uint8)

for c in possibleContours:
    cv2.rectangle(imgMask, pt1=(c['x'], c['y']), pt2=(c['w']+c['x'], c['h']+c['y']), color=(255, 255, 255), thickness=1)

plt.imshow(imgMask)
```

But we have many unecessary boxes...contours.     
Let's consider what a character looks like...about number plate.      
It consists of "### ####" in Korea.

```
def matchPlate():
```
