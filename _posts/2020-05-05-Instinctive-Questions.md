---
title: "Instinctive Questions"
date: 2020-05-05 13:08:00 +0900
categories: Computer DL
---

## Batch Size   
### Is batch-size must be even number? 배치 사이즈는 꼭 짝수여야할까?
from 1 to total dataset size 범위에서 가장 좋은 variance&generalization을 얻을 수 있는 batch size를 사용하면 되므로, 꼭 짝수가 아니어도 괜찮다.    
GPU의 memory size를 고려하고, parallelization&computation cost를 효율적으로 사용할 수 있는 batch size를 선택하는 것도 중요하다.    

## GPU   
### AMD vs nVIDIA?
because of easily installation of the graphics driver and using CUDA + cuDNN, nVIDIA is better for now.    

### 1 Good GPU vs Multiple Bad GPUs   
The performance of multiple bad gpus is worst than 1 good gpu.    
For using multiple gpus, you need to prepare much more space on your motherboard and bridge for connecting them, like as SLI, CrossFire.    
That bridge would decrease the performance while interconnecting multiple gpus and also would cause some system stability issues.    

### nVIDIA + Radeon 혼합하여 사용가능할까
SLI only work for 2 nVIDIA gpus, and CrossFire only work for 2 AMD gpus now.    
If they can connect each other, it would be working.    

### 1 Server with multiple GPUs, 여러 유저들이 GPU를 쉴틈없이 효율적으로 사용할 수 있는방법은?     


## Input   
### Input resolution, input image size에 따라 성능이 달라질까    
Input image size가 큰 high resolution input
### Input image의 Width X Height 크기를 다르게 하면 어떻게 될까   
### Input의 화질, Low quality image vs High quality image   
### 같은 이미지를 여러 개의 resolution으로 렌더링하면 영향이 있을까   
### Gray scale을 하고, 색반전해서 W/B -> B/W 로 하면 데이터갯수를 2배로 만들 수 있나?   
### CNN에서, channel을 압축하여 2D로 사용하면 어떨까?   
R1줄 G1줄 B1줄   
R2줄 G2줄 B2줄   
R3줄 G3줄 B3줄   



## Dataset   
### Could we know that how much dataset we need to train our model? 학습에 필요한 데이터셋의 크기를 어떻게 계산할까?   


## Untitled   
### 좋은 궁합의 recipe가 있을까 ex)Relu + Adam or Relu + AdamW   
### Neural Network의 whole equation을 구하고, 수식을 변환해서 간단하게 만들 수 없을까   
### learning rate 1 ~ 0.0000...1 까지 값을 넣고,같은 gradient 값을 내는 경우는 버림, 같은 기울기 값이므로 해가 아닐 확률이 높음?   
### learning rate의 값이 0.00... 인데, scale up 숫자를 사용하면 미분에 이득이 있을까      
### Drop Out할 때, Random Drop Out하면 각각의 경우에 성능이 달라질까 + 특정 Node의 성능이 좋다는 것을 알아낼 수 있을까
Network에서 더 좋은 성능을 내는 Neuron or Node 있을까?

### Domain randomization을 이용한 GAN    
### Test셋이 모수라면, test셋에 overfitting시키면 되지 않을까?   
하지만, test셋이 모수가 아닐 경우에는 성능이...    

### 램 꽂을때 2번&4번에 먼저 꼽는 이유
특정 cpu들의 경우, cpu가 second slot의 ram -> first slot의 ram 을 single-channel mode로 사용한다.
그렇기 때문에 2번 슬롯을 먼저 채우고, dual-channel mode를 사용하고자 할 때는 2번&4번 슬롯에 ram을 장착함으로써,
2->1, 4->3 dual-channel mode로 ram을 사용할 수 있게 된다.    
dual-channel mode에서 2&4 vs 1&3의 경우, performance 차이는 거의~아예 없다.    

### 논문을 보고 Network architecture를 github에 추가해주는 프로그램    
paper의 figure를 읽고, Network를 구성해주는 tool?    
image의 layer와 label을 dataset으로 만듦.    
ex) Conv2D 그림 or 글씨(OCR) -> Conv2D    
    Pooling 그림 or 글씨(OCR) -> Pooling    
    
### Training은 performance가 좋고, test에서는 inference가 빠른 모델은 어떻게 만들까?    
### Proper training for proper Anchor Box
### multiple optimizer
### multiple augmentations
