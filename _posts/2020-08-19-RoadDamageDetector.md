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

There are 3 countries such as Czech, India and Japan. And also four categories D00, D10, D20 and D40 about damage on the global roads.
We need to transform our VOC data format properly for specific object detection task.    
1. COCO format (for SimpleDet and MMDetection, TBD)    
2. JSON format (customizing myself in SimpleDet)    
3. YOLO format (YOLO family)    

Take option no.3 for now, make it as a YOLO format.    
dataset split... maybe K-fold then ensemble.    

1. Faster R-CNN    
2. EfficientDet    
3. YOLO v4    
4. YOLO v5    
5. DetectoRS       


There are 10 classes, but we need just four classes.    
```
#label_map = {"D00": 1, "D01":2, "D10": 3, "D11": 4, "D20": 5, "D40": 6, "D43": 7, "D44": 8, "D50": 9, "D0w0": 10}
label_map = {"D00": 1, "D10": 2, "D20": 3, "D40": 4} # what we need
```

1. Remove the dataset if it has useless label.    
2. Remove just that label with bbox in that dataset.    

Let's go with no.2-!    

We will use that datasets by using Pseudo-labeling for non-label datasets if we can.    



2. YOLO

Let's start from YOLOv3 to YOLOv5.    
There are several open sources and github for YOLOv3.
This is pjreddie's,
<https://pjreddie.com/darknet/yolo/>, <https://github.com/pjreddie/darknet>    

And this is Alexey's,
<https://github.com/AlexeyAB/darknet>    

I recommend to use Alexey's YOLOv3,    
because there are also YOLOv4 and pjreddie stopped updating.    

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
yolo family, v3, v4 and v5    
Japan: Japan, InJa, InJaCz    
India: India, InJa, InJaCz    
Czech: Czech, CzJa, InJaCz    

- For yolo, this is not easy to train well with mAP.    
So I decided to train the tiny model and tune it, then check out the result mAP.    
Anchor size will be changed, if it is not good with default values.    

metric: mAP,    
need specific strategy to split the dataset like train8:valid2,

To this comepetition, we need to check the result using F1 score with precision and recall.    

v5 looked like good than other versions,    

0.46: v4-tiny one model, Train with Czech + India + Japan at the same time.    
0.49: v4 CzInJa, v4 CzInJa, v5 Ja

yolov5 -> 0.5~

each single model: 0.51 or 0.52 

...

0.56: Ensemble using multiple models,    
Czech: Specific trategy    
India: Specific trategy    
Japan: Specific trategy   

single + other one


Last score: 7th in 120 teams.    
<https://rdd2020.sekilab.global/leaderboard/>
