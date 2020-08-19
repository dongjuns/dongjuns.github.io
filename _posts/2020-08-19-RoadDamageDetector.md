---
title: "Global Road Damage Detection Challenge 2020"
date: 2020-08-19 10:42:00 +0900
categories: detection
---

IEEE BingData 2020, Global Road Damage Detection Challenge 2020
<https://rdd2020.sekilab.global/data/>

## 0. Overview
Detecting the road damage.    

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

There are 3 countries and four categories about damage on the global roads.
