---
title: "Mathematics of ML"
date: 2019-05-02 14:33:00 +0900
categories: Math, ML, DL
use_math: true
---

Based on the Mathematicl principles in Machine Learning   
<https://www.edwith.org/2019090-aip2-advanced/joinLectures/22436>
- - -
## Concept of Machine Learning in the mathematics
머신러닝은 수많은 데이터들을 가지고, 각각의 class들을 잘 분류할 때 좋은 결과를 보여준다.   

feature와 class에 대한 정보를 담고있는 dataset을 평면위에 연이어 찍어보면,   
각각의 class들을 잘 나누는 function 혹은 plane을 찾을 수 있다.   
2D에서는 선으로, 3D에서는 면으로 Hyperplane을 구할 수 있으며,   
이것을 이용하여 기준을 정하고, feature들의 정보만 보고 class를 분류한다.   

그렇다면, hyperplane을 어떻게 찾을까?    
컴퓨터에게 굉장히 반복적인 계산을 시켜서 찾는다.   
먼저 임의로 초기 hyperplane을 정하고,    
Hyperplane과 class 1의 차이, hyperplane과 class 2의 차이를 계속해서 확인해나간다.   
Class 1과 class 2를 잘 분류하는, 차이가 가장 작은 hyperplane을 찾는 것이 목표다.    

어느정도의 문제까지는 인간이 계산기를 이용해서 hyperplance을 찾을 수 있을 것이다.     
하지만 class의 갯수가 많아질수록, feature의 갯수가 많아질수록,
hyperplane을 찾기 어려워지고 계산시간이 증가한다.    
그리고 이때부터 딥러닝(수학계의 노가다 김씨)의 진가가 발휘된다.    

Perceptron 기반의 구조를 만들고, 인풋이 들어가고    
layer마다 non-linear function을 사용하여 아웃풋을 얻어내고,   
이전 layer의 아웃풋이 다음 layer의 input이 되어 non-linear function...   
이와 같은 과정을 정해진 layer 수만큼 반복하여,    
일반적인 사람은 풀 수 없는,(또는 풀기 귀찮은, 또는 오래걸리는)    
Hyperplane을 찾아낸다.    
- - -

## Sparse Model    
Sparse: 희소한,    
Sparse vector & Sparse Matrix: Vector의 element가 거의 다 0임. 값을 갖고 있는 요소가 sparse, 희소하다.   
ex) sparse matrix = $[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]$

변수 1개인 Regression 문제를 생각해보면, $y = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + ... + a_n x^n$ 로 나타낼 수 있다.    
이 때 함수 y를 설명하는 f(x;a)가 몇개의 다항식에 연관되어있느냐를 고려해볼 수 있는데, sparse model은 계수가 0인 변수들을 사용하지 않는다.   
(1) 1~2개 정도의 x가 이용되어 y를 기술할 때 -> 전체적으로 f(x)가 약간 부실해보임.(언더피팅)    
(2) 적당한 정도의 x가 이용되어 y를 기술할 때 -> 전체적으로 f(x)가 y를 매우 잘 설명함,(Good 피팅) But 특정 포인트들에서는 조금 아쉬움.   
(3) 모든 x가 이용되어 y를 기술할 때 -> f(x)가 y 그 자체. 그러나, 매우 복잡한 모델을 갖게 되기 때문에 새로 들어오는 데이터셋들에 대해서는 정확하지 않을 수 있다.(오버피팅)    
$\rightarrow$ Sparse Model이 오버피팅에 대한 위험도를 감소시킬 수 있다. -> f(x)를 잡을 때는 적당히 sparse한 모델이 좋겠다!    

- - -

### 이득 of Sparse Modeling   
오버피팅을 줄이고, 어느정도 Regularization을 갖을 수 있다. 

(1) Sparsity -> Polynomial Function으로 solution이 나올 때, 계수가 거의 0이므로 Regularization -> 간단하다, 안복잡하다 -> Less overfitting -> Good prediction    
(2) Sparsity -> Variable Selection에서 계수가 거의 0이므로 사용하는 feature들이 적다 -> 모델에 대해서 설명하기가 용이하다. Model interpretability        
결국, 실제값 Y가 어떻게 나오는 지를 정확히 예측하는 함수를 만들고자 함임.
Y를 WX로 나타낼 수 있음. -> 사전행렬과 Sparse Vector의 곱으로 나타냄 ->Dictionary Matrix * Sparse Vector    
Min||Y - WX||^2 의 form을 나타냄.(Loss function의 냄새가 살짝 나고 있음)   

Problem : 수많은 사전행렬xSparse Matrix의 조합 중에서 어떤Sparse Matrix를 어떻게 잡을 것인가?   
-> Regularization term 추가.    

그래서 Min||Y - WX||^2 +r ||X|| 로 표현하게 되고, r은 람다. 가중치. 이게 커지면, Model의 성과보다는 희소값을 증가시키고자 함.   
||X|| 는 vertor Norm이고, "Length" of vector를 뜻한다.   
||X||_p = (Sum|x_i|^p)^(1/p) = (|x1|^p+|x2|^p+...+|xn|^p)^(1/p)

이 때 p의 값을 기준으로 L_p Norm을 정의할 수 있게 되고,   
p = 1, L1 Norm = (|x1|+|x2|+...+|xn|) : 벡터의 모든 요소들의 절대값의 합.    
p = 2, L2 Norm = (|x1|^2+|x2|^2+...+|xn|^2)^(1/2) : 벡터의 모든 요소들의 절대값의 제곱합의 제곱근.       
p = 무한 = max(x_i) : 벡터의 모든 요소들 중에서 가장 큰 값을 뽑아서 씀.  
p = 0, L0 Norm = number(Non-zero element) 벡터 내의 모든 요소들 중에서 0이 아닌 것들의 갯수를 값으로 씀.    
->벡터 X는 K-Sparse, K-희소이고, 만약 x = (0,0,1,0,3) 이라면, non-zero elements는 2개 이므로 L0 Norm 은 2, 2-Sparse.    


X의 elements가 0인 것들이 많을수록(Sparse), X에 대한 존재감이 작아지면서 자연스럽게 Regularization term의 값도 작아진다.    
그러므로, Sparse Vector X를 더더더더 Sparse하게 만들어서 equation의 값을 최소화 시킨다.
Sparse Modeling을 Loss function + Regularization term 으로 보겠음.   
Min(Loss + Regularization), Min(||Y - WX||^2 +r ||X||)
Loss function : Dictionary Matrix와 Sparse vector 로 만든 값이 실제값 Y랑 얼마나 비슷하냐?    
Regularization term : Sparse Vector와 Dictionary Maxtrix의 Sparsity를 평가.    
 

### Maxtrix Decomposition
행렬분해. Y = WX, 행렬 W와 X를 가지고 행렬 Y의 근사치를 나타냄.    
그리고 행렬 W와 X에 대한 constraints에 따라서 4가지 행렬 분해 방법이 있음.   

### (1) Sparse Coding (SC)
Y(mxn) = W(mxd)X(dxn)임. 이 때, Sparse Coding에서는 W가 m<d 여야 한다고 제약을 줌. W가 데이터의 갯수(m)보다 각각의 feauture들의 갯수(n)가 훨씬 많아야 하게 됨.  
그러면, Overcomplete 행렬 W가 있을텐데, 여기에다가 element가 거의 다 0 값인, 최대한 sparse vector를 곱해서 Y를 표현하자는 것임.   
그리고 제약 하나 더, Y = WX를 구할 때, ||X||가 작아야 함. 이 때에는 L0 Norm or L1 Norm 으로 사용함.   
하지만, D가 이미 overcomplete하기 때문에 X가 Y에 대해 그닥 의미가 없어지는 경우를 방지하기 위해서,    
||D||^2 <= C 제약을 통하여, D의 요소가 너무 커지는 것을 다시 막는다.

### (2) Principal Component Analysis (PCA)   
행렬 분해 방법이면서 Dimensionality Reduction Method.
포인트는 예측값의 변화가 가장 큰 벡터 u를 찾는 것, 그래서 variance가 고려된다.    
수십개의 feature가 있을 때, 이 feature들이 각각의 차원을 의미하고, 수많은 차원의 feauture로 범벅된 Y값에는,   
사실 모든 차원이 고려되야만 하는 것은 아닐 것이다. Y값을 잘 기술할 수 있는 적당한 feature들을 골라내는 것을 목표로 한다.    
feature들간의 공분산 행렬을 고려하며, feature1-feature2 간의 변동값이 얼마나 비슷한 지를 확인한다.   
그래서 공분산 행렬의 eigenvector와 eigenvalue를 계산하고, eigenvector에 대해 projection하는 작업을 반복해 나간다.    
feature들의 변동을 제대로 확인하기 위해서, feature값들에 대해, 그것들의 평균을 빼주어서 모든 feature들이 각각 평균값이 0이 되도록 맞춰주어야 한다.    
PCA는 Min||Y - WX||^2 +||D^TD -L||^2 + ||X^TX - Sum||^2 으로 나타낼 수 있고,
L2 Norm, D가 column full-rank, D와 X가 orthonormal matrix라는 제약을 갖는다.


### (3) Independent Component Analysis (ICA)   
Y : Observation Matrix, independent sources로부터 관측됨.   
W : mixing matrix, 혼합 행렬. 각 요소들이 independent source들의 혼합비율을 나타냄.    
X : source matrix. vector xi, xj가 independent.
observation matrix Y를 기반으로, source matrix X를 찾아냄.   
먼저, Whiten the data. feature들간 연관성이 없고, unit variance를 가지고 있는 데이터로 만들어줌.    
Mean Shift : Y = Y - E[Y] 로 해줌. Y 값에서 Y평균값 빼줘서, Y의 평균을 0으로 만듦.    
그리고 cov(Y)=E[YY^T]=WE[XX^T]W^T=WW^T 로 Covariance Matrix 하고,   
Eigen Analysis를 통해 UrU^T 고유벡터와 고유값 행렬을 얻음.    
그리고, whitening matrix Q와 obervation matrix Y를 곱해서, whtened data Y' 을 얻음.    
마지막으로, Y'를 optimization시켜서 데이터를 회전시키고, Source Matrix를 예측함.
-> Y를 Whitening, Y'을 optimization, Source를 예측.    
ICA에서는 L2 Norm을 제약으로 사용.


### (4) Non-negative Matrix Factorization (NMF)
Y(mxn) = W(mxd)X(dxn) 임. 포인트 Regularization은 Y,D,X 모두 음수가 아니어야 함.   
행렬 Y를 W와 X의 곱으로 표현해내며, 특정한 계수들로 표현할 수 있음.   
자연어 처리에서 유용, 데이터를 데이터의 일부들로 설명할 수 있다는 장점이 있음.   

- - -

### Regularized likelihood methods
regression은, y를 맞추기 위한 f(X;W) = x1w1 + x2w2 + ... + xnwn + error(bias) 를 찾는 것임.   
그래서, dataset이 주어지면 적절한 W값을 곱해서 예측값을 얻어내고, Y와 비교함.   
이 과정에서 예측값과 Y값의 차이를 최소로 하는 W를 찾는 것이 목표임.
Onjectives,
Least Square 최소제곱법 : W-hat_{LS} =  min(Sum(y - f(X;W)))^2, 제곱의 합을 줄이자!    
Maximum likelihood : W-hat_{ML} = max(Sum(logp(y|X;W))),    
Y가 관찰되었을 때, Y가 이렇게 관찰될 확률을 가장 크게 만드는 parameter를 찾자!     

Regularization을 쓰는 이유 : 모델이 너무 복잡해지지 않도록 하기 위해서.    
complex model에 쉽게 노출되어 있다, simple model이 설명하기 좋다. 

\theta^{hat} = argmin_{theta}(loss(y, f(X;theta)) + \lambda \psi(\theta))



(1) Lasso : L-1 regularization, Least Absolute Shrinkage and Selection Operator.
덜 중요한 feature들을 0으로 축소시켜서 sparse modeling 시킴. But, feautre >> data samples 인 경우에는 convex하지 않고 eigenfunction을 못 찾을 수 있음.

(2) Ridge : L-2 regularization

(3) Elastic Net : L-1 + L-2 regularization

(4) Group Lasso


### Causal : 인과  
수많은 feature로 이루어진 관계속에서 원인과 결과를 연결, cause & effect  
Association != Causation  
causal effect는, X가 Y를 야기할 때 X의 변화가 Y의 변화에 미치는 영향력을 뜻한다.  
Causal의 3단계 : Association 연관 < Intervention 조정 < Counterfactuals 반사실  

시계열 데이터에서
(1) Granger Causality : G -> X & G -> Y : causality 일 때, 이것을 X->Y : Granger causality 로 봄. 정확한 인과 관계를 찾는 것은 아님.  


