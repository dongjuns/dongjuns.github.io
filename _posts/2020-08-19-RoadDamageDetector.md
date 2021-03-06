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
# test1 set
wget -c https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/test2.tar.gz

tar xzvf train.tar.gz
tar xzvf test1.tar.gz
tar xzvf test2.tar.gz
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
So we need to do some pre-processing.
```
#label_map = {"D00": 1, "D01":2, "D10": 3, "D11": 4, "D20": 5, "D40": 6, "D43": 7, "D44": 8, "D50": 9, "D0w0": 10}
label_map = {"D00": 1, "D10": 2, "D20": 3, "D40": 4} # what we need
```

- Remove just that label with bbox in that dataset.    

Let's go do that thang-!    


## 2. YOLO
There are several open sources and github for YOLO family, let's gradually start from YOLOv3 to YOLOv5.    
First of all, v5 doesn't mean it updated from v4.    
There were different approaches to develop the YOLOv3,    
v4 and v5 both are just those approaches.    
But I think that naming is a little bit confusing.    

This is pjreddie's YOLOv3 repository,    
<https://pjreddie.com/darknet/yolo/>, <https://github.com/pjreddie/darknet>    

And this is Alexey's repository, <https://github.com/AlexeyAB/darknet>    

I recommend to use Alexey's thing,    
because it also support you about YOLOv4, and pjreddie stopped updating.    

When you try to use the YOLO with your gpu, you need to modify the Makefile.    
```
GPU=1
CUDA=1
OpenCV = 1
```
then re-try make again.    

If you get an error like below, you don't need to worry.    
```
/usr/bin/ld: skipping incompatible /usr/local/cudnn/v5/lib64/libcudnn.so when searching for -lcudnn
/usr/bin/ld: cannot find -lcudnn
collect2: error: ld returned 1 exit status
```
It means there is no libcudnn.so file in your cuda-cudnn path.    
Just copy your libcudnn.so to your cuda-cudnn path, like /usr/local/cuda/lib64/    

- Experiment    
yolo family, v3, v4 and v5    
Japan: Japan, InJa, InJaCz    
India: India, InJa, InJaCz    
Czech: Czech, CzJa, InJaCz    

1k, 2k, 3k,

- For yolo, this is not easy to train well with mAP.    
So I decided to train the tiny model and tune it, then check out the result mAP.    
Anchor size will be changed, if it is not good with default values.    

used metrics: mAP, F1 score    
We need specific strategy to split the dataset like train 80% : valid 20%,    

To this comepetition, we need to check the result using F1 score with precision and recall.    

v5 looked like good than other versions,    

0.46: v4-tiny one model, Train with Czech + India + Japan at the same time.    
0.49: v4 CzInJa, v4 CzInJa, v5 Ja

yolov5 -> 0.5~

each single model: 0.51 or 0.52, nms threshold, TTA

...

0.56: Ensemble using multiple models,    
Czech: Specific trategy    
India: Specific trategy    
Japan: Specific trategy   

Cross validation,
single + other one

Find strong one and mix it

0.58

some improvements changing some values 0.59

Last score: 7th in 120 teams.    
<https://rdd2020.sekilab.global/leaderboard/>

You can find more detailed information in this paper.

<https://www.researchgate.net/publication/345196850_Road_Damage_Detection_Using_YOLO_with_Smartphone_Images>    
DOI: <10.13140/RG.2.2.11013.58089>

If you want to get much more higher score,    
Make a detector with various resolution such as 416, 448, 480, ....    
Then ensemble them 10 or 20 models.    
You could get a score up to 0.67.    
