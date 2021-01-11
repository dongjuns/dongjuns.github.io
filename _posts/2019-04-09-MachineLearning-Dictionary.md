---
title: "Machine Learning Recipe & Dictionary"
date: 2019-04-09 17:27:00 +0900
categories: Machine_Learning
use_math: true
---


### Outlier, Novel data
Outlier, 이상한 값. 뭔가 어떠한 이유로 인해, distribution에서 나올 수 없는데, dataset에 있는 값.    
Outlier와 Noise는 다르다.    
Outlier의 경우, 패턴 내에서 특이한 양상을 보이기 때문에, 이런 특성은 도난 방지에도 사용할 수 있다.

### Novelty Detection
Global outlier: general dataset과 꽤 다른 모습을 나타냄. 얼마나 떨어져 있는 가를 측정해서 손쉽게 구별가능.    
Local outlier: 특정 부분에서만 다른 모습을 나타냄.    
Collective outlier: outlier가 아닌 정상적인 data인데, dataset 전체를 보면 차이가 심함.    
Ex) bug로 인해서, 같은 데이터가 수십번 반복생성.

### Regularization
Lasso: L1 norm

Ridge: L2 norm

ElasticNet: Lasso + Ridge

### AutoML


### Hyper-parameterization, Fine tuning
HyperparameteΩ: model을 training하기 전에, 사람이 먼저 정해줘야하는 parameters.    
hidden layer의 갯수, learning rate, batch size, epochs 등등.   
algorithm에 따라서 필요한 hyperparameter의 종류도 다를 수 있다.   
그리고 최적의 Hyperparameter 값을 찾는 것을 Hyper-parameterization or Fine tuning 이라고 하며,   
값을 계속 바꿔가며 찾는 try and error 방법, Bayesian optimization 방법 등등이 있다.

### Weight, 가중치
Hypothesis $H(x) = xW + b$ 일 때, data의 x값에 가중치 W를 이용하여,    
target값과 최대한 비슷한 값을 예측해내는 Hypothesis를 만듦.

### Cost Function, Loss function, 손실 함수, 비용 함수
Cost(W, b) = $\frac{1}{n}Sum(H(x) - y)^2$
Cost function: data와 hypothesis 간에 차이가 얼만큼 나는 지를 측정한다.   
Hypothesis = xW + b로 나타낼 수 있고, x가 데이터, W가 weight 가중치, b는 bias.    
model이 잘 맞는지 안 맞는지에 대해 cost function을 minimization 해가면서 minimum error를 가진 정확한 모델을 만들고자 함.   
그렇기 때문에, cost function is the optimization objective.   
그러면 cost function을 어떻게 줄이냐 ->  Gradient Descent.    

### Gradient Descent algorithm, Iterative Descent algorithm, 경사 하강법
Minimize(cost function)을 위한 방법. 미분을 이용한다.   
$W := W - r * \frac{a}{aW} \frac{1}{2m} Sum(xW - Y)^2$
$W := W - r * \frac{1}{2m} 2Sum(xW - Y)x$
$W := W - r * \frac{1}{m} Sum(xW - Y)x$
이때, cost function이 convex (밥그릇 모양 함수) 하지 않으면 optimize minimize가 잘 안된다, 잘못할 수도 있다.   
cost function을 minimize하기 위해, W값을 조절하고 cost를 줄이고자 하고, 계속 반복적으로 수행 = Iterative algorithm.    
최종적으로는 기울기가 0으로, 미분값이 0으로, function이 수렴하는 방향으로 찾아가고자 함.   
W와 b에 대해서 따로 편미분해서 cost function의 minimum을 찾는 방법을 사용하지 않는 이유는,   
실제로 계산할 때의 방정식은 굉장한 다항식으로 나오기 때문에, 각각의 모든 항에 대해서 편미분하는 것이 더 expensive하기 때문.   

### Sum Squared Error 
$SSE = sum(x - m)^{2}$,    
Regression 회귀분석에서 자주 쓰임. But, SSE is not convex. It means, SSE doesn't sure about global minimum.

### Cross Entropy
Cross Entropy = $(-x * log(m) - (1 - x) * log(1 - m))$    

Classification, Logistic Regression 에서 자주 쓰임

### Non-Linearity
- 데이터의 확률 분포를 알맞게 그려낼 수 있다면, 꽤 괜찮은 성능의 모델을 만들었다고 볼 수 있음.    
But, 데이터에 포함된 feature의 갯수가 많고 복잡한 차원으로 표현된다면 선형이 아닌 비선형으로 나타남.    
Multi-layer에서 모델을 activation function으로 나타내면, Non-linearity를 갖는 합성 함수의 성질을 갖게 됨.    


### Activation function, 활성 함수
- Sigmoid Function, Logistic Function
$z = f(x) = \frac{1}{(1 + e^{-x})}$   
값을 0~1 사이의 값으로 transfromation 해줌, 그렇기 때문에 결과를 확률처럼 이용할 수 있음.
safety zone 이상에서는 아웃풋이 0, 1로 saturate됨, gradient가 0으로 계산됨.    


- ReLU (Rectified Linear Unit)   
y = max(0, x), 아주 간단한 공식이기 떄문에 계산이 매우 쉽고 빠름.    


### Softmax
결과값을 0~1 사이의 값으로 바꿔줌.   
ex) 결과값이 (A, B, C) = (3.0, 1.2, 0.4) 였고, A로 분류했다면,    
(3.0, 1.2, 0.4)를 값 / Sum 으로 해줘서 -> (0.65, 0.26, 0.09) 로 맞춰줌. 이러면, 확률적으로 결과를 고려할 수 있게됨.


### One-Hot encoding
softmax 를 통해서 나온 0~1 숫자들을 하나의 class에 대해서만 값을 준다.    
softmax 후에 (0.65, 0.26, 0.09) -> One-Hot encoding 해서 (1, 0, 0)



### Regression
Continuous, 연속적인 값을 갖는 feature 들 간의 상관관계를 이용하여, Target과 Input features 사이의 correlation을 찾아내고자 함.    
주식 예측, 매출 예측, 온도 예측 등등

- Linear regression
Understand the correlation between the input features and target feature.   
쉽게 생각하면 2D x-y 좌표계에서 x,y 값들을 point out 한다.    
그러면, x-y 사이에 관련된 방정식을 찾아낼 수 있고, 이것을 바탕으로 Hypothesis를 세우고, x값이 주어졌을 때 y값을 예측해볼 수 있다.   
보통, feautres가 여러개인 Multiple linear regression의 경우가 많다.    
1D linear regression : Hypothesis $H(x1) = x1W1 + b$    
2D linear regression : Hypothesis $H(x1, x2) = x_1 W_1 + x_2 W_2 + b$   
이렇게 보았을 때, y에 관련된 x1, x2의 영향력을 볼 수 있고, 이것은 W1, W2를 통해 방정식에 반영된다.    
최종적으로 계산값에 대해서 Sigmoid function을 사용한다.    
Classification : 0 or 1   
Logistic regression : 0~1   

하지만, 주로 H(X) = XW 로 두고, matrix로 계산한다.   
그 이유는 실제로 데이터를 보았을 때,   
data1 = x_11, x_12, x_13, y_1     
data2 = x_21, x_22, x_23, y_2   
이런 식으로 값이 있을 것이고,   
X = (x_11, x_12, x_13)으로 두고, W = (w_1, w_2, w_3)에 대해서 matrix 연산을 하면 더 빠르게 계산할 수 있다.   


- SVM, Support Vector Machine regression

- Random Forest regression

- SGD, Stochastic Gradient Descent
optimizer. loss fucntion의 기울기가 0인 값을 찾아 Weight값을 optimize하는 역할.   
Gradient Descent를 반복적으로 수행하며 찾아간다.

- Bayesian regression

### Classification
dataset의 input features <-> target feature 사이의 연관성을 파악하여,   
data의 class를 맞추는 것이다.   
Ex) Spam or not spam, Cat or Dog, Disease or not    

- Logistic Regression
$H_{L}(x) = xW$   
Logistic regression은 Classification algorithm 중의 하나.    
비연속적인, discrete class들에 대해서 예측하는 용도로 사용.     
Ex) 개 or 고양이 / spam or not spam

It's not make the result as an 0 or 1, (so doesn't match to binary classification.)   
그래서 결과를 0 or 1로 만드는 방법이 필요하고, linear function 보다 정교한 cost function을 사용해야함.    
-> It's the Sigmoid Function or Logistic Function.    
$g(H_{L}(x)) = \frac{1}{(1 + e^{-x})}$    

그래서 결과값이 [0, 1] 사이의 확률값으로 나올 수 있게 된다.   

$H_{R}(x) = g(H_{L}(x)) $   
Logistic regression 에서 linear regression의 cost function SSE 같은 것을 cost function으로 그냥 써버릴경우에는,    
non-convex & 여러개의 local minimum을 가질 수 있음    
-> Local minima에 빠짐 & global minimum을 찾기 굉장히 어려움 & cost function minimize 힘듦.    

- SVM

- Random Forest

- DT, BDT: Decision Tree, Boosted Decision Tree


### Clustering
비슷한 feautres를 갖는 target value들을 Clustering. Unsupervised Learning에서 사용,   
only input features만 가지고 target value를 이해하는데에 사용한다.   
Ex) 토마토 in vegetable, Personal customers clustering

### CNN
Convolutional layer + Pooling layer를 relu와 Dropout으로 Non-linearty network를 만들고,     
fully connected-layer를 이용하여 ANN claasifier를 구성한다.
Convolutional layer를 learnable filter로 보면, image에 kernel filter matrix를 적용하여 feature maps를 뽑아내는 것으로 볼 수 있다.     
여기에 Pooling layer를 이용해서 Down sampling을 해주는데, 계산적인 cost를 줄이고 overfitting을 피할 수 있게 해준다.
결국 CNN은 Convolutional layer + Pooling layer로 image의 local feature 들을 뽑아내고, 이것으로 이미지의 전체적인 특성을 파악하는 것으로 볼 수 있다. 그렇기 때문에 filter의 갯수와 size가 매우 중요하다.     
Dropout을 이용하여 layer의 nodes을 트레이닝에서 randomly 사용하지 않음으로써 network를 regularization해주는 효과가 있고,     
generalization을 해주고 overfitting을 피할 수 있게 해준다. 당연히 test에서는 모든 network를 사용해준다.      

relu는 rectifier activation function, rectifier는 정류기! 물리 실험 시간에 사용했던 그 정류기... 특정 threshold 전에는 0, 그 이후에는 값이 인가된다.      
이 특성을 이용하여, input의 값이 x일 때, max(0, x) function을 이용하여 neural network에 non-linearity를 준다.     
Flatten layer가 있음으로 인해 feature maps를 1D vector로 flatten하고 이것을 이용하여 classification을 할 수 있게 된다.     
마지막으로, class에 대해 classification probability를 나타내주는 softmax를 이용한다.


