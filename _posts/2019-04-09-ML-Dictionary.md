---
title: "Machine Learning Recipe & Dictionary"
date: 2019-04-09 17:27:00 +0900
categories: Machine Learning
---


용어 정리



[1] Cost Function, Loss function
Sum Squared Error (SSE) = sum(x - m)^{2}
회귀분석에서 자주 쓰임. But, SSE is not convex. It means, SSE doesn't sure about global minimum.


Cross Entropy = (-x*log(m) - (1-x)*log(1-m))
Classification, Logistic Regression 에서 자주 쓰임



[2] Sigmoid Function, Logistic Function
z = 1 / (1 + e^{-x})

(1) Regression
Usually, In the Regression use the Cross Entropy.





Binary Classification

(2) Logistic Regression
H_{L}(x) = xW

It's not make the result as an 0 or 1, so doesn't match to binary classification.
그래서 결과를 0 or 1로 만드는 방법이 필요함.
-> It's the Sigmoid Function or Logistic Function.
g(H_{L}(x)) = 1 / (1 + e^{-2})

그래서 결과값을 [0, 1] 사이로 나오게 함.

H_{R}(x) = g(H_{L}(x))
