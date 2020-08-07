---
title: "Statistics Dictionary"
date: 2019-04-23 16:10:00 +0900
categories: Statistics
---

0.평균 average = sum(Xi) / n     
기대값 mean = sum(Pi*Xi), E[average] = mu

1.Probability vs Likelihood
Probability: 사건이 일어나기 전에 결과를 예측    
Likelihood: 일어난 사건을 이용하여 이전 상태를 확인    

확률 분포의 parameter를 추정할 때는 Likelihood를 많이 사용한다.    
예를 들어, mean과 sigma를 이용하는 Gaussian distribution에서 sigma값을 정확히 알고 있고 mean값은 모르는 경우라고 한다면,    
Likelihood function을 이용하여 Maximum Likelihood Estimation(MLE)를 갖는 mean값을 구하고, 해당 경우의 mean값이라고 고려할 수 있다.

이미 관측된 데이터 X를 이용하여, 모수 parameter theta에 대한 가능도를 구한다.    

2.Maximum Likelihood Estimation(MLE)
최대가능도추정, parameter에 대해 likelihood 가능도가 최대인 값을 구하는 것인데,
likelihood에 log를 씌워 computation cost를 줄이고,    
LogL(theta)를 미분해서 LogL(theta)이 최대가 되는, LogL(theta)의 미분 = 기울기가 0이 되는 값을 찾는다.


3.Parameter
모수, 데이터에 대한 확률 분포에 영향을 미친다. mean, sigma 등등.
특정 확률분포에 대해서 paramter에 해당하는 것들의 값을 알고 있다면, 해당 데이터가 가지고 있는 확률분포를 이해할 수 있다.    
