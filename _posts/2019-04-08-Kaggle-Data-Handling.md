---
title: "Data Science: Data Handling"
date: 2019-04-08 13:03:00 +0900
categories: Data Science
---

### Standard Process of data preprocessing
거의 케글 공식입니다.         
(1) 데이터 로드 (Load the dataset with Pandas)
```
train = pd.read_csv("nameOfTrainDataset.csv")
test = pd.read_csv("nameOfTestDataset.csv")
```
(2) 전체 데이터셋 확인 및 shape 체크 (Check the dataset)
```
train.head() # .head(행 갯수) 가능, head(10) -> 10개만 보여줌
train.shape  # (row, columns) 형태로 print out됨

test.head()
test.shape
```
(3) train set -> features + target column vector 들로 나누기
```
y_train = train['타겟 칼럼의 이름']
x_train = train.drop(columns=['타겟 칼럼의 이름']) #응용해서 버리고 싶은 columns 더 버리기 가능.
```

(4) Null값, Missing value 체크 (데이터에서 빵꾸 찾기)
```
dataset.isnull().describe()
dataset.isnull().any().describe()
```

dataset.fillna() function으로 filling 가능.
```
dataset.fillna(0) # 0으로 채우기
dataset.fillna(1) # 1로 채우기
values = {"featur1":0, "feature2":3}
dataset.fillna(value=values) # feature 이름에 따라서 지정된 values 로 채우기
```

Null value 버리기
```
dataset = dataset.drop(dataset.loc['버리려는feature'].isnull()].index)
```

그 외에 mean 값, 근처 값, 자주 나오는 값 등등 많은 방법이 있음.

(5) feature값 Normalization [0, 1]           
Normalization을 해줌으로써, 조명에 따른 영향을 덜 받을 수 있고, CNN이 더 빨리 Convergence할 수 있게 해준다.            
```
# image 데이터 기준, 픽셀 값이 0~255 까지면
x_train = x_train / 255.0
test = test / 255.0
```

+ 필요한 경우에 Scaling 가능.
Log transformation, Standardization, Min-Max Scaling 등등.

(6) Reshape
image data의 경우, csv file에서 1D vector로 저장되어있을 것이다.           
ex) 28x28 이미지의 경우, 784개의 feature를 가진 n개의 데이터이다.         
이것들을 다시 3D image 데이터로 reshape해준다. (가로x세로xcolor 채널: 3D, grayscale=1 or RGB=3)
```
x_train = x_train.values.reshape(데이터갯수인데 -1로 편하게 두기 가능, width pixel 갯수, height pixel 갯수, color channel 갯수)
test = test.values.reshape()
```

(6) Label one-hot encoding          
label distribution을 확률적으로 볼 수 있게 해줌.
```
y_train = to_categorical(y_tain, num_classese=label distribution 갯수)
# [0,0,1,0] 이런식으로 label encoding 해줌.
```

(7) trian + validation 나누기
```
sklearn.model_selection 의 train_test_split 기능을 사용한다.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1 아니면 0.2 정도, random_state=숫자) 
```
test_size를 잡아줄때, 0.2 처럼 20%를 나타내는 형식으로 넣지않고 정수값을 넣으면, 정수만큼의 데이터 개수를 split해준다.           
random_state는 reproducible results를 위하여 사용하며, 같은 데이터셋 같은 option 값으로 trian_test_split 해주면 같은 결과를 얻을 수 있다.         
random값을 얻어낼 때 random.seed(값)으로 볼 수 있겠다.
random_state를 사용하지 않으면, np.random number generator에 의해 split 결과가 달라진다.          
label distribution 비율을 지키면서 split하고 싶다면, stratify=True option을 추가로 사용한다.


### Data Science procedure
1. Data Check: Find the Null, NaN, non-reasonable data at the bunch of dataset.
2. Data Visualization: Make a plot which people can understand easily, using matplotlib, seaborn and pandas...etc.
3. Data Feature Engineering: Change the type or distribution of data, Make a much more good feature.

only for dataset, this is a start package.

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



### pandas + seaborn,

target class별로 갯수 확인가능.
```
g = sns.countplot(y_train)
y_train.value_conuts()
```

확률적인 분포를 보고 싶다면,
```
sns.distplot(column)
```

input variable과 target variable scatter 확인,
```
data = pd.concat([dataset['target'], dataset['feature']], axis=1) #axis=1 빼면 안됨.
data.plot.scatter(x='feature', y='target')
```


+@ 2D scatter plot에서 Outlier 찾았을 때, feature별로 sort_values 해서 찾거나 제거 가능.
```
dataset.sort_values(by='feature 이름', ascending=False)[:n]
dataset = dataset.drop(dataset[dataset['Id'] == 인덱스넘버].index)
```


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

and number of k variables multivariate correlation check,
```
k = 10
cols = corrmatrix.nlargest(k, 'target_name')['target_name'].index
cm = np.corrcoef(dataset[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, xticklabels=cols.values, yticklabels = cols.values)
plt.show()
```

인풋변수 별로 target이 어떻게 나오는 지에 대해서 바이올린 플랏도 확인가능,
```
g = sns.violinplot(y="target", x="input_variable", data=dataset, inner="quartile")
```

feature별로 Missing value 갯수와 Percentage 확인,
```
misiingTotal = df.isnull().sum().sort_values(ascending=False)
missingPercent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missingData = pd.concat([total, percent], axis=1, keys=['missingTotal', 'missingPercent'])
missingData.head()
```

### log transformation
혹시, variable distribution에 skewness가 있다면 log transformation을 해주어서 normal distribution의 shape을 얻을 수도 있다.

```
dataset['target_name'] = np.log(dataset['target_name'])
```


### Normalization
```
train = train / 255.0 #make it [0, 1] from [0, 255]
```

### Reshape
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
