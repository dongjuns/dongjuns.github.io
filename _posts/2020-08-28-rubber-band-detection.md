---
title: "WRS dataset, rubber band detection"
date: 2020-08-28 10:42:00 +0900
categories: detection
---

## 0. Overview
Detecting the rubber band within several objects    

## 1. Dataset
<https://github.com/RasmusHaugaard/wrs-data-collection>
object classes: 12

training set: 200
test set: 50

but it could be gotten much more images using domain randomization for WRS dataset    

## 2. Object Detection
EfficientDet D0~D7, D4 showed best mAP 0.866   

## 3. Get the bbox position of the rubber band    
3-1. Remove the color threshold for just orange rubber band.    
3-2. straight-forward solution: using blender simulation environment, set the camera paremeters and size, position and objects also.    

## 4. Object detection using real time input image    
