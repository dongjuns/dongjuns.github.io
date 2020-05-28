---
title: "Instinctive Questions"
date: 2020-05-05 13:08:00 +0900
categories: Computer DL
---

## Batch Size   
### 배치 사이즈는 왜 짝수로 사용할까   


## GPU   
### 1 Good GPU vs Multiple Bad GPUs   
### Nvidia + Radeon 혼합하여 사용가능할까   


## Input   
### Input image size에 따라 성능이 달라질까   
### Input size의 가로X세로 크기를 다르게 하면 어떻게 될까   
### 저화질 원본 이미지 vs Super Resolution 이미지   
### 같은 이미지를 여러 개의 resolution으로 렌더링하면 영향이 있을까   
### Gray scale을 하고, 색반전해서 W/B -> B/W 로 하면   
### CNN에서, channel을 압축하여 2D로 사용하면 어떨까?   
R1줄 G1줄 B1줄   
R2줄 G2줄 B2줄   
R3줄 G3줄 B3줄   



## Dataset   
### 학습에 필요한 데이터셋의 크기를 어떻게 알아낼까   



## Untitled   
### 좋은 궁합의 recipe가 있을까 ex)Relu + Adam or Relu + AdamW   
### Neural Network의 whole equation을 구하고, 수식을 변환해서 간단하게 만들 수 없을까   
### learning rate 1 ~ 0.0000...1 까지 값을 넣고,같은 gradient 값을 내는 경우는 버림, 같은 기울기 값이므로 해가 아닐 확률이 높음?   
### learning rate의 값이 0.00... 인데, scale up 숫자를 사용하면 미분에 이득이 있을까      
### Drop Out할 때, Random Drop Out하면 각각의 경우에 성능이 달라질까 + 특정 Node의 성능이 좋다는 것을 알아낼 수 있을까   
### Domain randomization을 이용한 GAN 
### Test셋이 모수라면, test셋에 overfitting시키면 되지 않을까?   
하지만, test셋이 모수가 아닐 경우에는 성능이 
### 램 꽂을때 2번 4번에 먼저 꼽는 이유   
2번 슬롯으로 들어가서 1번 슬롯으로 나가는 구조이기 때문에.   
