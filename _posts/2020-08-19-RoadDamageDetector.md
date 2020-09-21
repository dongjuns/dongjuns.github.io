---
title: "Global Road Damage Detection Challenge 2020"
date: 2020-08-19 10:42:00 +0900
categories: detection
---

IEEE BingData 2020, Global Road Damage Detection Challenge 2020    
<https://rdd2020.sekilab.global/data/>

## 0. Overview
Detecting the road damages    

## 1. Dataset
<https://github.com/sekilab/RoadDamageDetector>

We can directly download the dataset in that github, or using 'wget' in terminal.
```
mkdir RoadDamageDataset && cd RoadDamageDataset

# train set
wget -c https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/train.tar.gz

# test1 set
wget -c https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/test1.tar.gz

# + test2 set will be released on 10th, September

tar xzvf train.tar.gz
tar xzvf test1.tar.gz
```

then you could get a data structure like this.
```
(base) dongjun@dongjun-System-Product-Name:~/djplace/RoadDamageDataset$ tree -d
.
├── test1
│   ├── Czech
│   │   ├── annotations
│   │   │   └── xmls
│   │   └── images
│   ├── India
│   │   ├── annotations
│   │   │   └── xmls
│   │   └── images
│   └── Japan
│       ├── annotations
│       │   └── xmls
│       └── images
└── train
    ├── Czech
    │   ├── annotations
    │   │   └── xmls
    │   └── images
    ├── India
    │   ├── annotations
    │   │   └── xmls
    │   └── images
    └── Japan
        ├── annotations
        │   └── xmls
        └── images

26 directories

```

There are 3 countries Czech, India and Japan, also four categories D00, D10, D20 and D40 about damage on the global roads.
Transform to the proper format for training in object detection.    
1. to COCO format (for SimpleDet and MMDetection, TBD)    
2. to JSON format (customizing myself)    


Take option 2 for now, and then make it to COCO format.    
dataset split... maybe K-fold then ensemble.    
1. Faster R-CNN    
2. EfficientDet    
3. YOLO v4    
4. YOLO v5    
5. DetectoRS    

Mix-up, cut-mix, cut-mix & mix-up, mosaic    
WBF, Pseudo Labeling    
Multi-Scale Testing    


For dataset formatiing,    
There are 10 classes, but we need just four classes.    
```
#label_map = {"D00": 1, "D01":2, "D10": 3, "D11": 4, "D20": 5, "D40": 6, "D43": 7, "D44": 8, "D50": 9, "D0w0": 10}
label_map = {"D00": 1, "D10": 2, "D20": 3, "D40": 4} # what we need
```

1. Delete the dataset if it has useless label.    
2. Delete just that label with bbox in that dataset.    

Let's go with no.2-!    

We will use that datasets by using Pseudo-labeling for non-label datasets.    


2. Yolo
When you try to use the YOLO with your gpu, you need to modify the Makefile.
´´´
GPU=1
CUDA=1
OpenCV = 1
´´´
then re-try make.

if you get the error like this, don't worry.
´´´
/usr/bin/ld: skipping incompatible /usr/local/cudnn/v5/lib64/libcudnn.so when searching for -lcudnn
/usr/bin/ld: cannot find -lcudnn
collect2: error: ld returned 1 exit status
´´´
it means, there is no libcudnn.so file in your cuda-cudnn path.
just copy your libcudnn.so to your path, like /usr/local/cuda/lib64/    

- Experiment    
yolo family, v3 and v5    
Japan: Japan,InJa, InJaCz    
India: India, InJa, InJaCz    
Czech: Czech, CzJa, InJaCz    

EffDet family, 0 5    
Japan: Japan,InJa, InJaCz    
India: India, InJa, InJaCz    
Czech: Czech, CzJa, InJaCz    


Ensemble using multiple models,    

- For yolo, this is not easy to train well with mAP.    
so, I decided to train the tiny model and tune it, then check out the result mAP.    
Anchor size will be changed, if it is not good with default values.    

metric: mAP,    
need specific strategy to split the dataset like train8:valid2,
