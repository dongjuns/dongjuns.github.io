---
title: "Project 2: How many dataset we need?"
date: 2020-09-08 12:18:00 +0900
categories: Dataset DeepLearning
---

# Project2.How Many Dataset We Need?
It is very comfortable to make the model with a lot of dataset.    
But we already know, gathering enough dataset is very hard even we don't know that how many dataset we need.    

Dataset: MNIST fashion dataset    
dataset shape: (60000, 28, 28)    
dataset labels: (10, 60000),    
label 0:9, 6000 images by label    

Hypothesis: there is an optimal number of dataset for the model    
+@ checking the result of accuracy and loss,    
it would be gaussian distribution.    
so, we can be estimating that best performance.    


- Sampling the specific number of images by class    
- 100 trials for each experiments
- gaussian fitting for them, train accuray / validation accuracy

|Number of dataset by class | dataset split ratio|batch size|epoch| train mean | train std | validation mean | validation std |
|---------------------------|--------------------|----------|-----|------------|-----------|-----------------|----------------|
|                      10   |         5:5        |     1    |  1  |    0.25    |    0.05   |       0.47      |       0.08     |
|                      10   |         5:5        |     1    |  2  |    0.63    |    0.04   |       0.57      |       0.06     |
|                      10   |         5:5        |     1    |  3  |    0.74    |   -0.06   |       0.6       |       0.06     |
|                      10   |         5:5        |     1    |  4  |    0.81    |    0.07   |       0.63      |      -0.05     |
|                      10   |         5:5        |     1    |  5  |    0.87    |    0.06   |       0.64      |       0.05     |
|                      10   |         5:5        |     1    |  6  |    0.91    |    0.04   |       0.66      |       0.04     |
|                      10   |         5:5        |     1    |  7  |    0.96    |    0.03   |       0.68      |       0.01     |
|                      10   |         5:5        |     1    |  8  |    0.98    |    0.03   |       0.68      |       0.02     |
|                      10   |         5:5        |     2    |  1  |    0.22    |   -0.05   |       0.47      |      -0.08     |
|                      10   |         5:5        |     2    |  2  |    0.64    |   -0.04   |       0.58      |       0.06     |
|                      10   |         5:5        |     2    |  3  |    0.77    |    -0.05  |       0.62      |       0.05     |
|                      10   |         5:5        |     2    |  4  |    0.82    |    0.07   |       0.65      |       0.04     |
|                      10   |         5:5        |     2    |  5  |    0.91    |    0.05   |       0.65      |       0.04     |
|                      10   |         5:5        |     2    |  6  |    0.93    |    0.05   |       0.66      |       0.03     |
|                      10   |         5:5        |     2    |  7  |    0.96    |    0.03   |       0.68      |       0.01     |
|                      10   |         5:5        |     2    |  8  |    0.99    |    0.00   |       0.69      |      -0.03     |
