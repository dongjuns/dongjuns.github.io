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
                                  
imgThresholdBlur = cv2.GaussianBlur(imgThreshold, ksize=(5,5), sigmaX=0)                          
```

*adaptiveMethod, T(x,y) is related to the blockSizeXblockSize - C      
-> ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C     

*threshholdType, THRESH_BINARY or THRESH_BINACRY_INV


Build a mask (0, 0, 0, ..., 0) and extract the contours.      
It retrieves all the contours using the image of binary pixels.
```
imgMask = np.zeros((height, width, channel), dtype=np.uint8)

contours, _ = cv2.findContours(imgThresholdBlur,
                                  mode=cv2.RETR_LIST,
                                  method=cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(imgMask, contours=contours, contourIdx=-1, color=(255,255,255))

plt.figure(figsize=(12,12))
plt.imshow(imgMask)
```

then draw all the contour from saved contours in findContours.
```
contours = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)
    
    contours_dict.append({
        'contour':contour,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'cx':x+(w/2),
        'cy':y+(h/2)
    })
    
plt.imshow(temp_result)
```
