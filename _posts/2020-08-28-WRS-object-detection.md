---
title: "Object detection for World Robot Summit"
date: 2020-08-28 10:42:00 +0900
categories: detection
---

# TBD    
code repository    
<https://github.com/dongjuns/WRS_ObjectDetection>    

wrs_coco format    
README.MD    

pre-print    
training phase plot    
validation    
mAP    

code clean up and commit    
Decide the journal


## 0. Overview
Detecting the WRS objects for the robot arm to solve pick&place task    

## 1. Dataset
<https://github.com/RasmusHaugaard/wrs-data-collection>    
object classes: 12, 1920 x 1200 pixels, RGBA 4 channels, 250 images    
strategy: train 200 images, validation 50 images    

## 2. Object Detection
EfficientDet D0~D7    

## 3. Get the bbox position of the rubber band    
3-1. Remove the color threshold for just orange rubber band.    
3-2. straight-forward solution: using blender simulation environment, set the camera paremeters and size, position and objects also.    

## 4. Object detection using real time input image

## 5. Results    

Best result: EfficientDet-D4, mAP 0.866    
|Model          |obj1	|obj2	|obj3	|obj4	|obj5	|obj6	|obj7	|obj8	|obj9	|obj10 |obj11	|obj12	|All mAP	|FPS|
|---------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|-------|------|---------|---|
|EfficientDet-D0|	 0	|0.77	|0.5	|0.63	|0.86	|0.77	|0.24	|0.19	|0.12	|0.07	 |0.02	 |0.007	|0.349	|61|
|EfficientDet-D1|0.38 |0.98	|0.66	|0.89	|0.94	|0.91	|0.63	|0.59	|0.52	|0.26	 |0.06	 |0.005	|0.569	|50|
|EfficientDet-D2|0.87 |0.99	|0.68	|0.96	|0.95	|0.92	|0.77	|0.71	|0.85	|0.68	 |0.43	 |0.14	|0.746	|44|
|EfficientDet-D3|0.97 |0.99	|0.77	|0.96	|0.97	|0.94	|0.83	|0.79	|0.85	|0.76	 |0.79	 |0.59	|0.853	|31|
|EfficientDet-D4|0.99 |1 	  |0.48	|0.96	|0.99	|0.97	|0.88	|0.81	|0.9	|0.83	 |0.86	 |0.74	|0.866	|21|
|EfficientDet-D5|	1	  |0.76	|0.54	|0.62	|0.97	|0.98	|0.47	|0.69	|0.91	|0.79	 |0.82	 |0.68	|0.769	|13|
|EfficientDet-D6|0.99 |0.64	|0.51	|0.49	|0.99	|0.98	|0.35	|0.77	|0.92	|0.78	 |0.77	 |0.68	|0.74	  |10|
|EfficientDet-D7|0.99	|0.99	|0.73	|0.54	|0.98	|0.95	|0.38	|0.78	|0.91	|0.82	 |0.83	 |0.66	|0.797	|7|
