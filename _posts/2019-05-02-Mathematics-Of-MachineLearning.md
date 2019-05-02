---
title: "Mathematics of ML"
date: 2019-05-02 14:33:00 +0900
categories: Machine Learning
---

Based on the Mathematicl principles in Machine Learning   
<https://www.edwith.org/2019090-aip2-advanced/joinLectures/22436

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

