---
title: "YOLO Family"
date: 2020-11-16 12:54:00 +0900
categories: YOLO
---

This covers YOLO family from v1 to v5.    
Then we are going to see how to use YOLO v3, v4 and v5.    

# YOLOv1
You Only Look Once: Unified, Real-Time Object Detection (2016, <https://arxiv.org/abs/1506.02640>)    
10 pages (2 pages are for references)    
<http://pjreddie.com/yolo/>    

기존에는 classifier를 사용해서 detection 문제를 해결했고, YOLO 논문에서는 new approach로써 detection 문제를 regression으로 해결하는 방법을 제안했다.    
Unified single convolutional neural network를 이용하여 end-to-end로 최적화를 하고 45 ~ 155 FPS를 지원한다.    
R-CNN과 같은 region proposal-based method는 이미지 안에서 potenstial bounding box를 만들고 난 후에 classifier를 통해 object detection을 한다.    

Input image resizes to 448 x 448.    
Image를 7x7 그리드로 나누고, 각각의 그리드 셀에서 2개의 b-box(Cx,Cy,w,h,confidence)와 class probabilities를 regression한다.    
Total prediction tensor는 7x7x{2x(Cx,Cy,w,h,confidence)+class probabilities}가 되고,    
S=7, B=2, number of class=20 일 때 7x7x(2x5 + 20)이 되며,    
total prediction boxes는 7x7x2 = 98 개다.    



# YOLOv2
Input image resizes to 416 x 416.    
Total prediction tensor는 13x13xB(Cx,Cy,w,h,confidence,class probabilities)가 되고,    
S=13, B=5, number of class=80 일 때 13x13x5(Cx,Cy,w,h,confidence,class probabilities)이 되며,    
total prediction boxes는 13x13x5 = 845 개다.  


# YOLOv3
YOLOv3: An Incremental Improvement (2018, <https://arxiv.org/abs/1804.02767>)    
6 pages (2 pages are for references)    
<https://pjreddie.com/darknet/yolo/>    

Input image resizes to 416 x 416.    
3 scale(/32, /16, /8)에 대해서 각각 3개의 bbox를 예측하기 때문에    
scale 1: 416/32 = 13, 13x13 grid에서 3개의 bbox = 507    
scale 2: 416/16 = 26, 26x26 grid에서 3개의 bbox = 2,028    
scale 3: 416/8 = 52, 52x52 grid에서 3개의 bbox = 8,112    
= 10,647 total anchor boxes    

각각의 anchor box는 (Cx,Cy,w,h,confidence,class probabilities)를 예측한다.    


# YOLOv5

### 1. Dataset    
First of all, you need labeling your custom images.    
If you have your familiar tool, do that thang.    
Or I recommend you labelbox <https://labelbox.com/>.    
You can do labeling on website without installation any labeling tool or github labeling repository.    

1. Make your new project and fill out some information, then attach your dataset.    

2. In attach a dataset, click 'upload new dataset' and really upload it.    

3. In customize your label editor - Select existing - Editor,    
click the tab 'Configure' then you could add your object!    
Click the 'Add object', then enter class name.    
Don't forget to change the option to Bounding box, if you want to do the object detection.    
Click the 'Confirm' and 'Complete setup'.    
Then you could see the tab 'Start labeling', also you could invite your friend.    


### 2. Architecture    
Backbone: CSPNet    
PANet    
Anchors    
Loss
