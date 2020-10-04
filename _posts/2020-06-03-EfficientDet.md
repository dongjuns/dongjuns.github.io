---
title: "EfficientNet"
date: 2020-06-24 11:37:00 +0900
categories: simpledet EfficientDet
---

# EfficientNet

### For parameter calculation

```
# EfficientNetParameterCalculator.py
a = 1.2
b = 1.1
r = 1.15

a_list = []
b_list = []
r_list = []

phi_list = [0, 0.5, 1, 2, 3.5, 5, 6, 7] # phi value of 0, 1, 2, 3, 4, 5, 6, 7

for phi in phi_list:
    new_a = pow(a, phi)
    new_b = pow(b, phi)
    new_r = pow(r, phi)
    a_list.append(new_a)
    b_list.append(new_b)
    r_list.append(new_r)
```


For channel of each stage i
```
channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
new_channels = []

b_length = len(b_list)
c_length = len(channels)

for b_num in range(b_length):
    for c_num in range(c_length):
        new_channel = b_list[b_num] * channels[c_num]
        new_channels.append(new_channel)
```

For depth of each stage i
```
layers = [1, 1, 2, 2, 3, 3, 4, 1, 1]
new_layers = []

a_length = len(a_list)
l_length = len(layers)

for a_num in range(a_length):
    for l_num in range(l_length):
        new_layer = a_list[a_num] * layers[l_num]
        new_layers.append(new_layer)
```

### modify the value of parameters for proper iteration
(1) batch size
```
...
class General:
    log_frequency = 10
    name = __name__.rsplit("/")[-1].rsplit(".")[-1]
    #batch_image = 8 if is_train else 1
    batch_image = 2 if is_train else 1
    fp16 = True
    loader_worker = 8
```

(2) epochs
```
...
class schedule:
    #mult = 6
    mult = 1
    begin_epoch = 0
    end_epoch = 6 * mult
```

Table for training from scratch
- batch size: 2 or 4 or 6 or 8,    
- optimizer: lr= , increasing rate: lr / iteration,        
- epochs:    
- top_N: 1000, 2000, ...,    
- NMS: 0.5 or 0.6    


# EfficientDet
for object detection,    
Backbone: EfficientNetBX(sharable parameters)    
Neck: BiFPN    
Head: N-class classification subnet + Bounding box regression subnet    


BiFPN: for different weight, can use fast normalization fusion.    

먼저, NAS-FPN을 이용해서 결과를 측정하고, Bi-FPN도 같이 수정해나간다.    

EfficientDet에 EfficientNet이 FPN과 연결되는 형태는,    
EfficientNet의 Stage와 관련이 있으며 특정 Stage에서 Input resolution, output resolution이 달라지는 것을 이용해서,    
다양한 resolution의 feature map을 사용하면 피라미드 모양의 Feature Pyramid Networks를 구성할 수 있다.
           input     output
Stage1에서 224X224 -> 112X112,    P1 feature maps, output size is 1/2 of input resolution    
Stage2에서 112X112 -> 112X112,    
Stage3에서 112X112 -> 56X56,    P2 feature maps, output size is 1/4 of input resolution    
Stage4에서 56X56 -> 28X28,    
Stage5에서 28X28 -> 28X28,    P3 feature maps, output size is 1/8 of input resolution    
Stage6에서 28X28 -> 14X14,    P4 fature maps, output size is 1/16 of input resolution    
Stage7에서 14X14 -> 7X7,    P5 feature maps, output size is 1/32 of input resolution    
Stage8에서 7X7 -> 7X7,    
Stage9에서 7X7 -> 3X3    


P6 feature maps, output size is 1/64 of input resolution    
P7 feature maps, output size is 1/128 of input resolution    

P6과 P7은 P5를 이용하여 얻어낸다.    

P6 feature maps, output size is 1/2 of P5 resolution (2X2 kernel로 stride 2)    
P7 feature maps, output size is 1/4 of P5 resolution (4X4 kernel로 stride 4)    

GPU 메모리의 제한(12GB)으로, EfficientNetB4까지만 트레이닝 가능.    

# To do list    
- Imagenet 1K for using pre-trained weight    
- COCO 2014 30class for using pre-trained weight    
- Dropout    
- Fast Normalization Fusion in BiFPN    
- From D0 ~ D7, compare the results of them    
- Transfer learning or Fine-tuning from COCO dataset to WRS dataset    


# Fine-tune
bbox_conv1234_weight, bbox_conv1234_bias, bbox_pred_weight, bbox_pred_bias    
cls_conv1234_weight, cls_conv1234_bias, cls_pred_weight, cls_pred_bias    
python으로 mxnet 임포트해서, pre-trained params 불러와서 classifier, regressor 부분 지워주고,    
nd.save 로 저장한다음에 pretrain_model로 보내서 detection_train.py

Anchor size 조절해야하고, Freeze 얼마만큼 할 것인지?    
Stage1만 가져왔을때 결과 0.857    


KFold then Ensemble
PyTorch multiple GPU
WBF    
Data Augmentation, CutMix, MixUp, Insect augmentation    
use Learning scheduler 

<https://github.com/rwightman/efficientdet-pytorch>



# tensorflow style
<https://github.com/google/automl>    
install them and then train.    

strategy:    
batch_size: 8, 4, 2    
num_epochs: 500, 1000    
num_examples_per_epch: 40, 80, 100, 200    
Anchor scale: TBD    
label_smoothing: TBD    
Augmentation: TBD    
