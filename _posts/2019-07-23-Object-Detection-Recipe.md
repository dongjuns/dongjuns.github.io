---
title: "Object detection recipe"
date: 2019-07-23 14:35 +0900
categories: Object_detection
---

# 1.Prepare or download the dataset    
Format style, Amount of dataset and labels, Labeling quality, Bbox quality    

# 2.Choose the framework
(1)PyTorch, TensorFlow with keras, mxnet    
(1-1)MMDetection, Detectron, SimpleDet    

# 3.Augmentation
DataGenerator, Albumentation, Cut-mix, Mix-up, Mosaic, Insect...    

# 4.Train the models
Faster R-CNN, EfficientDet, Yolo, Recent SOTA model    

# (Pseudo Labeling)

# 5.Ensemble

# 6.TTA


```
### 1. Introduction
### 2. Data preparation
## 2.1 Load data
## 2.2 Check for null and missing values
## 2.3 Normalization
## 2.4 Reshape
## 2.5 Label encoding
## 2.6 Split training and valdiation set
### 3. CNN
## 3.1 Define the model
## 3.2 Set the optimizer and annealer
## 3.3 Data augmentation
### 4. Evaluate the model
## 4.1 Training and validation curves
## 4.2 Confusion matrix
### 5. Prediction and submition
## 5.1 Predict and Submit results
```
