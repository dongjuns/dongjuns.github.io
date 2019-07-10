---
title: "Data Science : Data Handling"
date: 2019-04-08 13:03:00 +0900
categories: Data Science
---

### Data Science procedure
1. Data Check: Find the Null, NaN, non-reasonable data at the bunch of dataset.
2. Data Visualization: Make a plot which people can understand easily, using matplotlib, seaborn and pandas...etc.
3. Data Feature Engineering: Change the type or distribution of data, Make a much more good feature.

only for dataset, This is a start package.

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
```
import pandas as pd
```

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

If you want to check the label information,
```
trainSet["label_name"].value_counts()
```

data = pd.read_csv("파일이름"), data.head(), data.info(), data.describe(), data["variable"].value_counts() 만 기억하자.


## pandas + seaborn,

input variable 2개 두고, target 분포 확인가능.

```
tmp = data.drop("whatYouWannaDrop", axis=1)
g = sns.pairplot(tmp, hue="target", markers="+")
plt.show()
```

for making correlation heatmap,
```
sns.heatmap(data.corr(), annot=True) #annot show you the number of correlation of 2-variables
plt.show()
```

인풋변수 별로 target이 어떻게 나오는 지에 대해서 바이올린 플랏도 확인가능,
```
g = sns.violinplot(y="target", x="input_variable", data=dataset, inner="quartile")
```

# Normalization
```
train = train / 255.0 #make it [0, 1] from [0, 255]
```

# Reshape
Even, we can do "reshape" using pandas,
```
train = train.values.reshape(-1,가로,세로,채널(컬러))
```

- - -

### matplotlib
```
import matplotlib.pyplot as plt
%matplotlib inline
```

drawing a plot is super easy,
```
plt.show(x, y)
```

or if you wanna look at the real image,
```
plt.imshow(train)
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
