---
title: "Data Science : Data Handling"
date: 2019-04-08 13:03:00 +0900
categories: Data Science
---

### Data Science procedure
1. Data Check: Find the Null, NaN, non-reasonable data at the bunch of dataset.
2. Data Visualization: Make a plot which people can understand easily, using matplotlib, seaborn and pandas...etc.
3. Data Feature Engineering: Change the type or distribution of data, Make a much more good feature.

only for dataset, This is the start package.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.styple.use('seaborn')
sns.set(font_size=2.5)

%matplotlib inline
```

### PANDAS
read & load the csv datset
```
pd.read_csv('path/name.csv')
```


Declare the training set & test set of dataset
```
trainSet = pd.read_csv('path/name.csv')
testSet = pd.read_csv('path/name.csv')
```
and to read the values and columns and raws,
head() & describe() are enough.
```
trainSet.head() #first 5 values will be shown.
trainSet.describe() #more statistical approach for see the dataset
```

easy wat to check the Null check in python,
```
trainSet.agg(lambda x: sum(x.isnull()) / x.shape[0])
```

and we can see the multiple columns at the same time,
```
trainSet[['column1', 'column2']].groupby(['column1'], as_index=True).count()
```
also pandas support the good option 'crosstab'
```
pd.crosstab(trainSet['column1'], trainSet['column2'], margins=True).style.background_gradient()
```

- - -
### Numpy
```
import numpy as np
a = np.array([1,2,3,4,5])
a[-1]

b = np.array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
])
b[-1]
b[1]
b[:, -1]
b[:, 1]
b[0:2, :]
```
1행x5열, 1rowx5columns, (1, 5) 짜리 array a,   
3행x4열, 3rowx4columns, (3, 4) 짜리 array b.    
b[행, 열] 임. 

- - -

[1] Classification
- Binary Classification : Classify the data to 0 or 1, True or False.
(1) Titanic: Machine Learning from Disaster
Predict the survival passenger on the titanic.


- Boosted Decision Tree (BDT)
(1) Top quark mass measuremnt


[2] Regression
- Regression
(1) House Prices: Advanced Regression Techniques

(2) Posco AI 사내식당 식사인원 예측

- Logistic Regression

- - -
