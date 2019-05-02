---
title: "Mathematics of ML"
date: 2019-05-02 14:33:00 +0900
categories: Machine Learning
---

Based on the Mathematicl principles in Machine Learning   
<https://www.edwith.org/2019090-aip2-advanced/joinLectures/22436

### Concept of Machine Learning in the mathematics
머신러닝은 수많은 데이터들을 가지고, 각각의 class들을 잘 분류하기 위해 쓴다.   

feature와 class에 대한 정보를 담고있는 dataset을 평면위에 주르륵 찍어보면,   
각각의 class들을 잘 나누는 function 혹은 plane을 찾을 수 있다.
2D에서는 선으로, 3D에서는 면으로 Hyperplane을 구할 수 있으며,   
이것을 이용하여 기준을 정하고, feature들의 정보만 보고 class를 분류하는 것이다.   

Hyperplane을 어떻게 찾느냐?    
결국 컴퓨터에게 수학적인 노가다를 시키는 것이다.   
Hyperplane을 일단 정하고,    
Hyperplane부터 클래스1과, Hyperplane과 클래스2와의 차이를 계속해서 확인해나간다.   
클래스1과 클래스2를 잘 분류하면서, 차이가 적은 Hyperplane을 찾을 때까지 계속 노가다.    

그리고, 클래스의 갯수가 많아질수록, feature의 갯수가 많아질수록,
Hyperplane을 찾기 어려워지고 까다로워진다.    
이때부터 머신러닝이 사용하는 수학적 노가다의 진가가 발휘된다.    
Perceptron을 기반으로, 인풋을 넣고    
layer마다 non-linear한 function들을 계속해서 사용하여 아웃풋을 다시 넣고 반복하여,    
사람의 직관과 계산으로 풀 수 없는, 또는 풀기 귀찮은, 또는 오래걸리는    
하지만 끝내주는 Hyperplane을 찾는 것이다.    

- - -

### Sparse Model    
Sparse : 희소한, Sparse vector & Sparse Matrix : Element가 거의 다 0임. 값을 가지고 있는 요소가 희소함.    
이득 of Sparse Modeling
(1) Sparsity -> Polynomial Function으로 solution이 나올 때, 계수가 거의 0이므로 Regularization -> 간단하다, 안복잡하다 -> Less overfitting -> Good prediction
(2) Sparsity -> Variable Selection에서 계수가 거의 0이므로 사용하는 feature들이 적다 -> 모델에 대해서 설명하기가 용이하다. Model interpretability    
결국, 실제값 Y가 어떻게 나오는 지를 정확히 예측하는 함수를 만들고자 함임.
Y를 WX로 나타낼 수 있음. -> 사전행렬과 Sparse Vector의 곱으로 나타냄 ->Dictionary Matrix * Sparse Vector    

Problem : 그러면 어떻게 Sparse vector를 어떻게 잡을 것인가? -> Regularization과 Norm의 등장.   
Norm : Length of the vector, 벡터의 길이. ||x||_{p} = (Sum|x_{i}|^{p})^{1/p}으로 나타냄.    
이 때, p값이 0이냐 1이냐 2이냐 ... 무한대이냐로 L-$ Norm 정함.    
p = 1, L1 Norm : 벡터의 모든 요소들의 절대값의 합.    
p = 2, L2 Norm : 벡터의 모든 요소들의 절대값의 제곱합의 제곱근.   
p = 무한 : 벡터의 모든 요소들 중에서 가장 큰 값.   
p = 0, L0 Norm : 벡터 내의 모든 요소들 중에서 0이 아닌 것들의 갯수를 값으로 씀.   
->벡터의 L0 Norm이 K보다 작으면? 벡터 X는 K-Sparse, K-희소.   

자, Sparse Modeling을 Loss function + Regularization term 으로 보겠음.   
Min(Loss + Regularization)
Loss function : Dictionary Matrix와 Sparse vector 로 만든 값이 실제값 Y랑 얼마나 비슷하냐?    
Regularization term : Sparse Vector와 Dictionary Maxtrix의 Sparsity를 평가.    
Min((||Y - WX||^2) + r||X||), r은 람다. 가중치. 이게 커지면, Model의 성과보다는 희소값을 증가시키고자 함.

